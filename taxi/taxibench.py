import argparse
import glob
import io
import re
import subprocess
import sys

import mysql.connector

# Load database reporting functions
import report

omnisciExecutable = "build/bin/omnisql"
taxiTripsDirectory = "/localdisk/work/trips_x*.csv"

command1DropTableTrips = "drop table taxitestdb;"
command2ImportCSV = "COPY taxitestdb FROM '%s' WITH (header='false');"

timingRegexpRegexp = re.compile(
    r"Execution time: (\d+) ms, Total time: (\d+) ms", flags=re.MULTILINE
)
exceptionRegexpRegexp = re.compile("Exception: .*", flags=re.MULTILINE)

omnisciCmdLine = ["-q", "omnisci", "-u", "admin", "-p", "HyperInteractive"]
benchmarksCode = [
    """
\\timing
SELECT cab_type,
       count(*)
FROM taxitestdb
GROUP BY cab_type;
    """,
    """
\\timing
SELECT passenger_count,
       avg(total_amount)
FROM taxitestdb
GROUP BY passenger_count;
    """,
    """
\\timing
SELECT passenger_count,
       extract(year from pickup_datetime) AS pickup_year,
       count(*)
FROM taxitestdb
GROUP BY passenger_count,
         pickup_year;
    """,
    """
\\timing
SELECT passenger_count,
       extract(year from pickup_datetime) AS pickup_year,
       cast(trip_distance as int) AS distance,
       count(*) AS the_count
FROM taxitestdb
GROUP BY passenger_count,
         pickup_year,
         distance
ORDER BY pickup_year,
         the_count desc;
""",
]

tripsCreateTableOriginal = """
CREATE TABLE taxitestdb (
    trip_id                 INTEGER,
    vendor_id               VARCHAR(3) ENCODING DICT,

    pickup_datetime         TIMESTAMP,

    dropoff_datetime        TIMESTAMP,
    store_and_fwd_flag      VARCHAR(1) ENCODING DICT,
    rate_code_id            SMALLINT,
    pickup_longitude        DECIMAL(14,2),
    pickup_latitude         DECIMAL(14,2),
    dropoff_longitude       DECIMAL(14,2),
    dropoff_latitude        DECIMAL(14,2),
    passenger_count         SMALLINT,
    trip_distance           DECIMAL(14,2),
    fare_amount             DECIMAL(14,2),
    extra                   DECIMAL(14,2),
    mta_tax                 DECIMAL(14,2),
    tip_amount              DECIMAL(14,2),
    tolls_amount            DECIMAL(14,2),
    ehail_fee               DECIMAL(14,2),
    improvement_surcharge   DECIMAL(14,2),
    total_amount            DECIMAL(14,2),
    payment_type            VARCHAR(3) ENCODING DICT,
    trip_type               SMALLINT,
    pickup                  VARCHAR(50) ENCODING DICT,
    dropoff                 VARCHAR(50) ENCODING DICT,

    cab_type                VARCHAR(6) ENCODING DICT,

    precipitation           SMALLINT,
    snow_depth              SMALLINT,
    snowfall                SMALLINT,
    max_temperature         SMALLINT,
    min_temperature         SMALLINT,
    average_wind_speed      SMALLINT,

    pickup_nyct2010_gid     SMALLINT,
    pickup_ctlabel          VARCHAR(10) ENCODING DICT,
    pickup_borocode         SMALLINT,
    pickup_boroname         VARCHAR(13) ENCODING DICT,
    pickup_ct2010           VARCHAR(6) ENCODING DICT,
    pickup_boroct2010       VARCHAR(7) ENCODING DICT,
    pickup_cdeligibil       VARCHAR(1) ENCODING DICT,
    pickup_ntacode          VARCHAR(4) ENCODING DICT,
    pickup_ntaname          VARCHAR(56) ENCODING DICT,
    pickup_puma             VARCHAR(4) ENCODING DICT,

    dropoff_nyct2010_gid    SMALLINT,
    dropoff_ctlabel         VARCHAR(10) ENCODING DICT,
    dropoff_borocode        SMALLINT,
    dropoff_boroname        VARCHAR(13) ENCODING DICT,
    dropoff_ct2010          VARCHAR(6)  ENCODING DICT,
    dropoff_boroct2010      VARCHAR(7)  ENCODING DICT,
    dropoff_cdeligibil      VARCHAR(1)  ENCODING DICT,
    dropoff_ntacode         VARCHAR(4)  ENCODING DICT,
    dropoff_ntaname         VARCHAR(56) ENCODING DICT,
    dropoff_puma            VARCHAR(4)  ENCODING DICT
) WITH (FRAGMENT_SIZE=%d);
"""

tripsCreateTableFSI = """
CREATE TABLE taxitestdb (
    trip_id BIGINT,
    vendor_id TEXT ENCODING NONE,
    pickup_datetime TIMESTAMP,
    dropoff_datetime TIMESTAMP,
    store_and_fwd_flag TEXT ENCODING NONE,
    rate_code_id SMALLINT,
    pickup_longitude DOUBLE,
    pickup_latitude DOUBLE,
    dropoff_longitude DOUBLE,
    dropoff_latitude DOUBLE,
    passenger_count SMALLINT,
    trip_distance DOUBLE,
    fare_amount DOUBLE,
    extra DOUBLE,
    mta_tax DOUBLE,
    tip_amount DOUBLE,
    tolls_amount DOUBLE,
    ehail_fee DOUBLE,
    improvement_surcharge DOUBLE,
    total_amount DOUBLE,
    payment_type TEXT ENCODING NONE,
    trip_type TINYINT,
    pickup TEXT ENCODING NONE,
    dropoff TEXT ENCODING NONE,
    cab_type TEXT,
    precipitation DOUBLE,
    snow_depth SMALLINT,
    snowfall DOUBLE,
    max_temperature SMALLINT,
    min_temperature SMALLINT,
    average_wind_speed DOUBLE,
    pickup_nyct2010_gid BIGINT,
    pickup_ctlabel DOUBLE,
    pickup_borocode INT,
    pickup_boroname TEXT ENCODING NONE,
    pickup_ct2010 INT,
    pickup_boroct2010 INT,
    pickup_cdeligibil TEXT ENCODING NONE,
    pickup_ntacode TEXT ENCODING NONE,
    pickup_ntaname TEXT ENCODING NONE,
    pickup_puma INT,
    dropoff_nyct2010_gid BIGINT,
    dropoff_ctlabel DOUBLE,
    dropoff_borocode BIGINT,
    dropoff_boroname TEXT ENCODING NONE,
    dropoff_ct2010 INT,
    dropoff_boroct2010 INT,
    dropoff_cdeligibil TEXT ENCODING NONE,
    dropoff_ntacode TEXT ENCODING NONE,
    dropoff_ntaname TEXT ENCODING NONE,
    dropoff_puma INT
) WITH (FRAGMENT_SIZE=%d, STORAGE_TYPE='CSV:%s');
"""


def getErrorLine(text):
    # Try looking for an exception first
    exMatch = re.findall(exceptionRegexpRegexp, text)
    if exMatch:
        return exMatch

    # Return last non-emptry string from text buffer
    buf = io.StringIO(text)
    for line in buf:
        line = line.strip()
        if line != "":
            errStr = line
    return errStr


def testme():
    for benchNumber, benchString in enumerate(benchmarksCode, start=1):
        print(benchNumber, ":", getErrorLine(benchString))

    teststr1 = """
this is a test
Exception: Sorting the result would be too slow
this is a test
"""
    teststr2 = """
this is a test
Exception: Sorting the result would be too slow
Exception: Some other exception
this is a test
"""
    teststr3 = """
this is a test1

this is a test2

this is a test3

"""
    print("test1:", getErrorLine(teststr1))
    print("test2:", getErrorLine(teststr2))
    print("test3:", getErrorLine(teststr3))
    sys.exit()


parser = argparse.ArgumentParser(description="Run NY Taxi benchmark using omnisql client")

parser.add_argument(
    "-fs",
    action="append",
    type=int,
    help="Fragment size to use for created table. Multiple values are allowed and encouraged.",
)
parser.add_argument("-e", default=omnisciExecutable, help='Path to executable "omnisql"')
parser.add_argument(
    "-ct",
    action="store_true",
    help="Use CREATE TABLE WITH (STORAGE_TYPE='CSV:trips.csv'). KEEP IN MIND that currently it is possible to load JUST ONE CSV file with this statement, so join all data into one big file, so -df value has no effect.",
)
parser.add_argument(
    "-df", default=1, type=int, help="Number of datafiles to input into database for processing"
)
parser.add_argument(
    "-dp", default=taxiTripsDirectory, help="Wildcard pattern of datafiles that should be loaded"
)
parser.add_argument(
    "-dnd",
    action="store_true",
    help="Do not delete old table. KEEP IN MIND that in this case -fs values have no effect because table is taken from previous runs.",
)
parser.add_argument(
    "-dni",
    action="store_true",
    help="Do not create new table and import any data from CSV files. KEEP IN MIND that in this case -fs values have no effect because table is taken from previous runs.",
)
parser.add_argument(
    "-t",
    default=5,
    type=int,
    help="Number of times to run every benchmark. Best result is selected",
)
parser.add_argument(
    "-sco", action="store_true", help="Show commands (that delete and create table) output"
)
parser.add_argument("-sbo", action="store_true", help="Show benchmarks output")
parser.add_argument("-r", default="report.csv", help="Report file name")
parser.add_argument("-test", action="store_true", help="Run tests")
parser.add_argument(
    "-port",
    default=62074,
    type=int,
    help="TCP port that omnisql client should use to connect to server",
)

parser.add_argument("-db-server", default="localhost", help="Host name of MySQL server")
parser.add_argument("-db-port", default=3306, type=int, help="Port number of MySQL server")
parser.add_argument(
    "-db-user",
    default="",
    help="Username to use to connect to MySQL database. If user name is specified, script attempts to store results in MySQL database using other -db-* parameters.",
)
parser.add_argument(
    "-db-pass", default="omniscidb", help="Password to use to connect to MySQL database"
)
parser.add_argument(
    "-db-name", default="omniscidb", help="MySQL database to use to store benchmark results"
)

parser.add_argument(
    "-commit",
    default="1234567890123456789012345678901234567890",
    help="Commit hash to use to record this benchmark results",
)

args = parser.parse_args()

if args.test:
    testme()

if args.df <= 0:
    print("Bad number of data files specified", args.df)
    sys.exit(1)

if args.t < 1:
    print("Bad number of iterations specified", args.t)

omnisciCmdLine = [args.e] + omnisciCmdLine + ["--port", str(args.port)]

db_reporter = None
if args.db_user != "":
    print("Connecting to database")
    db = mysql.connector.connect(
        host=args.db_server,
        port=args.db_port,
        user=args.db_user,
        passwd=args.db_pass,
        db=args.db_name,
    )
    db_reporter = report.DbReport(
        db,
        "taxibench",
        {
            "FilesNumber": "INT UNSIGNED NOT NULL",
            "FragmentSize": "BIGINT UNSIGNED NOT NULL",
            "BenchName": "VARCHAR(500) NOT NULL",
            "BestExecTimeMS": "BIGINT UNSIGNED",
            "BestTotalTimeMS": "BIGINT UNSIGNED",
        },
        {"ScriptName": "taxibench.py", "CommitHash": args.commit},
    )

for fs in args.fs:
    print("RUNNING WITH FRAGMENT SIZE", fs)
    # Delete old table
    if not args.dnd:
        print("Deleting taxitestdb old database")
        try:
            process = subprocess.Popen(
                omnisciCmdLine,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE,
            )
            output = process.communicate(command1DropTableTrips.encode())
        except OSError as err:
            print("Failed to start", omnisciCmdLine, err)
        if args.sco:
            print(str(output[0].strip().decode()))
        print("Command returned", process.returncode)

    dataFilesNumber = 0
    # Create table and import data
    if not args.dni:
        dataFileNames = sorted(glob.glob(args.dp))
        if len(dataFileNames) == 0:
            print("Could not find any data files matching", args.dp)
            sys.exit(2)

        if args.ct:
            # Foreign storage interface import with CREATE TABLE
            dataFilesNumber = 1
            print(
                "Creating new table taxitestdb with fragment size",
                fs,
                "and data file",
                dataFileNames[0],
            )
            createTableStr = tripsCreateTableFSI % (fs, dataFileNames[0])
            try:
                process = subprocess.Popen(
                    omnisciCmdLine,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.PIPE,
                )
                output = process.communicate(createTableStr.encode())
            except OSError as err:
                print("Failed to start", omnisciCmdLine, err)
            if args.sco:
                print(str(output[0].strip().decode()))
            print("Command returned", process.returncode)
        else:
            # Import using COPY
            # Create new table
            print("Creating new table taxitestdb with fragment size", fs)
            createTableStr = tripsCreateTableOriginal % fs
            try:
                process = subprocess.Popen(
                    omnisciCmdLine,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.PIPE,
                )
                output = process.communicate(createTableStr.encode())
            except OSError as err:
                print("Failed to start", omnisciCmdLine, err)
            if args.sco:
                print(str(output[0].strip().decode()))
            print("Command returned", process.returncode)
            # Datafiles import
            dataFilesNumber = len(dataFileNames[: args.df])
            for df in dataFileNames[: args.df]:
                print("Importing datafile", df)
                copyStr = command2ImportCSV % df
                try:
                    process = subprocess.Popen(
                        omnisciCmdLine,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.PIPE,
                    )
                    output = process.communicate(copyStr.encode())
                except OSError as err:
                    print("Failed to start", omnisciCmdLine, err)
                if args.sco:
                    print(str(output[0].strip().decode()))
                print("Command returned", process.returncode)

    # Benchmarks
    try:
        with open(args.r, "w") as report:
            for benchNumber, benchString in enumerate(benchmarksCode, start=1):
                bestExecTime = float("inf")
                bestTotalTime = float("inf")
                errstr = ""
                for iii in range(1, args.t + 1):
                    print("Running benchmark number", benchNumber, "Iteration number", iii)
                    try:
                        process = subprocess.Popen(
                            omnisciCmdLine,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            stdin=subprocess.PIPE,
                        )
                        output = str(process.communicate(benchString.encode())[0].strip().decode())
                    except OSError as err:
                        print("Failed to start", omnisciCmdLine, err)
                    if args.sbo:
                        print(output)
                    print("Command returned", process.returncode)
                    execTime = float("inf")
                    totalTime = float("inf")
                    if process.returncode == 0:
                        matches = re.search(timingRegexpRegexp, output).groups()
                        if len(matches) == 2:
                            execTime = int(matches[0])
                            totalTime = int(matches[1])
                            print("Iteration", iii, "exec time", execTime, "total time", totalTime)
                        else:
                            print("Failed to parse command output:", output)
                            errstr = getErrorLine(output)
                    if bestExecTime > execTime:
                        bestExecTime = execTime
                    if bestTotalTime > totalTime:
                        bestTotalTime = totalTime
                print(
                    "BENCHMARK",
                    benchNumber,
                    "exec time",
                    bestExecTime,
                    "total time",
                    bestTotalTime,
                )
                print(
                    dataFilesNumber,
                    ",",
                    fs,
                    ",",
                    benchNumber,
                    ",",
                    bestExecTime,
                    ",",
                    bestTotalTime,
                    ",",
                    errstr,
                    "\n",
                    file=report,
                    sep="",
                    end="",
                    flush=True,
                )
                if db_reporter is not None:
                    db_reporter.submit(
                        {
                            "FilesNumber": dataFilesNumber,
                            "FragmentSize": fs,
                            "BenchName": str(benchNumber),
                            "BestExecTimeMS": bestExecTime,
                            "BestTotalTimeMS": bestTotalTime,
                        }
                    )
    except IOError as err:
        print("Failed writing report file", args.r, err)
