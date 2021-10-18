1. TeamCity host should have OmnisSciDB dependencies installed. Either
by using `scripts/mapd-deps-ubuntu.sh` ([see here for other
distros](https://github.com/omnisci/omniscidb/wiki/OmniSciDB-Dependencies))
or downloading and installing binaries from omnisci site.
2. User should have high limit for open files. Using `ulimit -n 10000`
in `~/.bashrc` is a good idea.
3. The following python3 packages should be installed using `pip`:
pymapd, braceexpand, mysql-connector-python.
4. User should have permissions to connect to MySQL host and insert
records there. Modify test scripts with user credentials and MySQL
server host name.
5. User should have installed conda or miniconda. Modin tests and benchmarks
run through run_modin_test.py script which initially creates environment with
combined requirements from ([ibis](https://github.com/ibis-project/ibis/blob/master/ci/requirements-3.7-dev.yml))
and your personal requirements, default is ci_requirements.yml.
All subsequent tasks such as build, test or benchmarks running are being done in created conda env.