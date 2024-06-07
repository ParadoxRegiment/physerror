import sys
import subprocess

reqs_dict = {
    'numpy': 'numpy',
    'pandas[excel]' : ('pandas', 'xlsxwriter', 'openpyxl'),
    'matplotlib' : 'matplotlib',
    'tk' : 'tk',
    'scipy' : 'scipy',
    'seaborn' : 'seaborn',
    'inquirer' : 'inquirer'
}

def _install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def _check_install():
    reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
    
    return installed_packages

def user_install():
    installed_pkgs = _check_install()

    for i in range(len(installed_pkgs)):
        installed_pkgs[i] = installed_pkgs[i].lower()

    warn_title = f"{'READ FULL MESSAGE BEFORE ANSWERING': ^65}\n"
    warn_msg = ("This file is used to check for installed Python libraries and install\n"
                "packages that are not already installed. The packages are not malicious\n"
                "in any way and can easily be found by searching their names online.\n"
                "However, if you are uncomfortable with this file installing the packages,\n"
                "please answer 'No' below.\n")
    print(warn_title)
    print(warn_msg)
    print(f"Required packages: {list(reqs_dict.keys())}")

    user_veri = input("If you would like to install the packages, type and enter 'Yes'. Otherwise type and enter 'No': ")
    print("")

    match user_veri.lower():
        case 'yes' | 'y':
            for req in reqs_dict:
                if type(reqs_dict[req]) == str:
                    if reqs_dict[req] not in installed_pkgs:
                        print(f"Installing {req}...\n")
                        _install(req)
                    else:
                        print(f"{req} already installed, continuing...")
                else:
                    for subreq in reqs_dict[req]:
                        if subreq not in installed_pkgs:
                            print(f"Installing {req}...\n")
                            _install(req)
                        else:
                            print(f"{req} - {subreq} already installed, continuing...")
        case _:
            sys.exit("User has cancelled installation.\n")

    print("\nRequired packages check complete.")
    installed_packages = _check_install()
    print(f"\nInstalled packages w/ dependencies:\n{installed_packages}\n")
    
if __name__ == "__main__":
    user_install()