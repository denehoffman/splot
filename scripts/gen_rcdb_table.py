import os

import numpy as np
import rcdb


def main():
    connection = os.environ.get("RCDB_CONNECTION")
    if not connection:
        connection = "mysql://rcdb@hallddb.jlab.org/rcdb"
    db = rcdb.RCDBProvider(connection)
    query = "@status_approved and (@is_production or @is_2018production or @is_dirc_production)"
    run_periods = [(30000, 40000), (40000, 50000), (50000, 60000), (60000, 80000)]
    total_table = []
    for run_min, run_max in run_periods:
        rcdb_runs = db.select_runs(query, run_min=run_min, run_max=run_max)
        rcdb_table = np.array(
            rcdb_runs.get_values(["polarization_angle"], insert_run_number=True),
            dtype=int,
        )
        total_table.append(rcdb_table)
    total_table = np.concatenate(total_table)
    np.savetxt("rcdb", total_table, delimiter="\t", fmt="%d")


if __name__ == "__main__":
    main()
