import ctypes
import json
import multiprocessing

import numpy as np
import pytest
from scipy import sparse

import lightgbm as lgb
from lightgbm.basic import _LIB, _c_str, _safe_call

from .utils import find_random_open_port


def _construct_distributed_dataset(data, params, machines, port, output):
    _safe_call(_LIB.LGBM_NetworkInit(_c_str(machines), ctypes.c_int(port), ctypes.c_int(1), ctypes.c_int(2)))
    try:
        lgb.Dataset(data, params=params)._dump_text(output)
    finally:
        _safe_call(_LIB.LGBM_NetworkFree())


def _train_with_missing_categorical_feature(
    data, label, categorical_feature, params, *, machines, port, test_data, output
):
    _safe_call(_LIB.LGBM_NetworkInit(_c_str(machines), ctypes.c_int(port), ctypes.c_int(1), ctypes.c_int(2)))
    try:
        dataset = lgb.Dataset(data, label=label, categorical_feature=[categorical_feature], params=params).construct()
        # Bins come from the owner, where this feature has no valid categories.
        assert dataset.feature_num_bin(categorical_feature) == 0
        booster = lgb.train(params, dataset, num_boost_round=5)
        assert booster.current_iteration() == 5
        np.save(output, booster.predict(test_data))
    finally:
        _safe_call(_LIB.LGBM_NetworkFree())


@pytest.mark.parametrize("input_type", ["dense", "csr", "file"])
@pytest.mark.parametrize("per_feature_max_bin", [False, True])
@pytest.mark.parametrize("forced_bins", [False, True])
def test_distributed_bin_feature_indices(tmp_path, input_type, per_feature_max_bin, forced_bins):
    # Both workers receive identical data so that sampling cannot explain any
    # difference from local binning. Worker 1 owns features 2 and 3, whose
    # ranges and configuration differ from features 0 and 1.
    values = np.arange(1, 129, dtype=np.float64)
    data = np.column_stack([values + 1000 * feature for feature in range(4)])
    params = {
        "num_threads": 1,
        "max_bin": 7,
        "min_data_in_bin": 1,
        "min_data_in_leaf": 1,
        "feature_pre_filter": False,
        "enable_bundle": False,
        "pre_partition": True,
        "verbosity": -1,
    }
    if per_feature_max_bin:
        params["max_bin_by_feature"] = [4, 5, 6, 7]
    if forced_bins:
        bounds = [
            {"feature": feature, "bin_upper_bound": [1000 * feature + 32.5, 1000 * feature + 96.5]}
            for feature in range(4)
        ]
        bounds_file = tmp_path / "forced_bins.json"
        bounds_file.write_text(json.dumps(bounds))
        params["forcedbins_filename"] = str(bounds_file)

    if input_type == "csr":
        data = sparse.csr_matrix(data)
    elif input_type == "file":
        data_file = tmp_path / "data.csv"
        np.savetxt(data_file, np.column_stack([np.zeros(len(values)), data]), delimiter=",")
        data = str(data_file)

    expected_file = tmp_path / "local.txt"
    lgb.Dataset(data, params=params)._dump_text(expected_file)
    expected = expected_file.read_text()

    ports = {find_random_open_port()}
    while len(ports) < 2:
        ports.add(find_random_open_port())
    machines = ",".join(f"127.0.0.1:{port}" for port in ports)
    context = multiprocessing.get_context("spawn")
    workers = []
    try:
        for rank, port in enumerate(ports):
            # Windows adapter enumeration does not include the loopback address.
            # Set the rank explicitly for these local workers.
            worker = context.Process(
                target=_construct_distributed_dataset,
                args=(data, params, f"rank={rank},{machines}", port, tmp_path / f"rank{rank}.txt"),
            )
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join(timeout=30)
            assert not worker.is_alive(), "Distributed Dataset construction timed out"
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)

    for rank in range(2):
        assert (tmp_path / f"rank{rank}.txt").read_text() == expected


@pytest.mark.parametrize("input_type", ["dense", "file"])
@pytest.mark.parametrize("missing_type", ["nan", "negative"])
@pytest.mark.parametrize("owner_rank", [0, 1])
def test_distributed_all_missing_categorical_feature(tmp_path, input_type, missing_type, owner_rank):
    # Four features give each worker two bin mappers. Exercise both owners,
    # including a missing category mapper after a nontrivial numeric mapper.
    categorical_feature = 2 * owner_rank + 1
    values = np.arange(1, 65, dtype=np.float64)
    test_data = np.column_stack([values, values % 3 + 1, values % 7, values % 5 + 1])
    label = 0.5 * values
    params = {
        "objective": "regression",
        "tree_learner": "data",
        "num_machines": 2,
        "num_threads": 1,
        "pre_partition": True,
        "label_column": 0,
        "min_data_in_leaf": 2,
        "num_leaves": 7,
        "force_col_wise": True,
        "enable_bundle": False,
        "verbosity": -1,
    }

    partitions = []
    for rank in range(2):
        data = test_data.copy()
        if rank == owner_rank:
            data[:, categorical_feature] = np.nan if missing_type == "nan" else -(values % 3 + 1)
        if input_type == "file":
            data_file = tmp_path / f"data{rank}.csv"
            np.savetxt(data_file, np.column_stack([label, data]), delimiter=",")
            data = str(data_file)
        partitions.append(data)

    ports = {find_random_open_port()}
    while len(ports) < 2:
        ports.add(find_random_open_port())
    machines = ",".join(f"127.0.0.1:{port}" for port in ports)
    context = multiprocessing.get_context("spawn")
    workers = []
    try:
        for rank, port in enumerate(ports):
            worker = context.Process(
                target=_train_with_missing_categorical_feature,
                args=(
                    partitions[rank],
                    label if input_type == "dense" else None,
                    categorical_feature,
                    params,
                ),
                kwargs={
                    "machines": f"rank={rank},{machines}",
                    "port": port,
                    "test_data": test_data,
                    "output": tmp_path / f"predictions{rank}.npy",
                },
            )
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join(timeout=30)
            assert not worker.is_alive(), "Distributed training with an all-missing categorical feature timed out"
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=5)

    predictions = [np.load(tmp_path / f"predictions{rank}.npy") for rank in range(2)]
    for prediction in predictions:
        assert prediction.shape == label.shape
        assert np.isfinite(prediction).all()
        assert np.ptp(prediction) > 0
    np.testing.assert_allclose(predictions[0], predictions[1], rtol=1e-12, atol=1e-12)
