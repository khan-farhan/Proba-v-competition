import os
from subprocess import call


def downloadData():
    print("Downloading...")

    if not os.path.exists("../Data"):
        call( 'mkdir ../Data' , shell=True )
        os.chdir(os.getcwd() + "/../Data")
        call('wget "https://kelvins.esa.int/media/competitions/proba-v-super-resolution/probav_data.zip"',shell=True)
        print("Downloading done.\n")

    else:
        print("Dataset already downloaded. Did not download twice.\n")
        return

    print("Extracting...")

    call( 'unzip -nq "probav_data.zip"', shell=True )
    print("Extracting successfully done.")
    print("Data is in ../Data directory")
    call('rm "probav_data.zip"',shell = True)
    os.chdir(os.getcwd() + "/../Codes")

    return

if __name__ == "__main__":
    downloadData()