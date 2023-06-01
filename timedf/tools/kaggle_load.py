import webbrowser

import zipfile


class KaggleAccessProblem(Exception):
    pass


def _download_zip(competition_name, local_dir, reload, rules_url):
    """Download zip file with all competition data. Can raise one of 2 errors:
    1. No user configuration found. User needs to have accurate kaggle API configuration
    2. No access to competiton data. User needs to accept competition rules.

    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi, ApiException
    except OSError as e:
        url = "https://www.kaggle.com/docs/api"
        print(e)
        print(
            f"Could not authenticate with kaggle api, please read about kaggle API authentification {url}"
        )
        webbrowser.open(url, new=0, autoraise=True)
        raise KaggleAccessProblem

    api = KaggleApi()
    api.authenticate()

    try:
        api.competition_download_files(competition_name, force=reload, path=local_dir, quiet=False)
    except ApiException as my_error:
        if my_error.reason == "Forbidden":
            print(
                "You don't have access to competition data, check that:\n"
                f"You've accepted competition rules at {rules_url}\n"
            )
            webbrowser.open(rules_url, new=0, autoraise=True)
            raise KaggleAccessProblem
        else:
            raise


def download_dataset(competition_name, local_dir, reload, rules_url):
    while True:
        try:
            _download_zip(competition_name, local_dir, reload, rules_url)
            break
        except KaggleAccessProblem:
            command = input(
                "There was a problem with access, described above.\n"
                'Please solve it and enter "y", or enter anything else to exit [y/n]:'
            )
            if command.lower() != "y":
                print("Exiting...")
                exit(1)
            else:
                print("\nTrying to load the data again...\n")

    print("Unzipping file...")
    with zipfile.ZipFile(local_dir / f"{competition_name}.zip", "r") as zip_ref:
        zip_ref.extractall(local_dir)
