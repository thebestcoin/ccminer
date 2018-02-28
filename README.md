The Bestcoin miner, ccMiner release for The Bestcoin
====================================================

CUDA-based miner for Lyra2REv2 The Bestcoin mining using 8x32 ASIC-resistant matrix (with higher memory requirement)

License
-------

See LICENSE.txt for additional info

Usage
-----

    ccminer -d <YOUR_nVIDIA_DEVICE_NUMBER> -o stratum+tcp://pool-mainnet1.thebestcoin.io:38802 -u <PUT_YOUR_TBC_ADDRESS_HERE> -p x

OR

    ccminer -d <YOUR_nVIDIA_DEVICE_NUMBER> -o stratum+tcp://pool-mainnet2.thebestcoin.io:38802 -u <PUT_YOUR_TBC_ADDRESS_HERE> -p x

OR

    ccminer -d <YOUR_nVIDIA_DEVICE_NUMBER> -o stratum+tcp://pool-testnet1.thebestcoin.io:38804 -u <PUT_YOUR_TBC_ADDRESS_HERE> -p x

OR

    ccminer -d <YOUR_nVIDIA_DEVICE_NUMBER> -o stratum+tcp://pool-testnet2.thebestcoin.io:38804 -u <PUT_YOUR_TBC_ADDRESS_HERE> -p x

(visit https://github.com/thebestcoin/thebestcoin/releases to obtain The Bestcoin wallet and get wallet address)

ccminer
-------

Based on Christian Buchner's &amp; Christian H.'s CUDA project
based on the Fork by tpruvot@github with X14,X15,X17,WHIRL,Blake256 and LYRA2 support , and some others, check the [README.txt](README.txt)
Reforked and optimized by sp-hash@github and KlausT@github 

SP-HASH: BTC donation address: 1CTiNJyoUmbdMRACtteRWXhGqtSETYd6Vd

A part of the recent algos were originally written by [djm34](https://github.com/djm34).

This variant was tested and built on Linux (ubuntu server 14.04) and VStudio 2013 on Windows 7.

Note that the x86 releases are generally faster than x64 ones on Windows.

About source code dependencies
------------------------------

This project requires some libraries to be built :

- OpenSSL (prebuilt for win)

- Curl (prebuilt for win)

- pthreads (prebuilt for win)

The tree now contains recent prebuilt openssl and curl .lib for both x86 and x64 platforms (windows).

To rebuild them, you need to clone this repository and its submodules :
    git clone https://github.com/peters/curl-for-windows.git compat/curl-for-windows

There is also a [Tutorial for windows](http://cudamining.co.uk/url/tutorials/id/3) on [CudaMining](http://cudamining.co.uk) website.

Build on Windows
----------------

Requirements:
* Windows 10, 8, 7, ... or whatever
* Visual Studio 2013 (There are known issues on VS2015, so be careful)
* CUDA Toolkit 9.0

OpenSSL, Curl and pthreads are already prebuilt in the project, as it stated above.  
Ensure you've selected the "Release" build configuration and Win32 platform (you may try x64 also).
In the Project Options, ensure the "Code Generation" option supports your video card.

Run "Build solution".  
Go to "Release" folder (or "x64\Release" if you've chosen x64 build). Check the program working:

    ccminer.exe --benchmark

Build on linux
--------------

Requirements:
* Ubuntu 16.04
* CUDA Toolkit 9.0
    ```
    sudo apt-get install autotools-dev automake pkg-config libtool libssl-dev libcurl4-openssl-dev
    ./autogen.sh
    ./configure
    ./build.sh
    echo $?
    ```

If the last command outputs "0", the build completed successfully. Run the command to check if everything is fine:

    ./ccminer --benchmark

Build on Linux using Docker nvidia/cuda image
---------------------------------------------

Requirements:

* Docker
* 2 Gb image download

Build the image

    ./docker-build.sh

Build the project

    ./docker-run.sh bash autogen.sh
    ./docker-run.sh ./configure
    ./docker-run.sh ./build.sh

Check the project

    ./docker-run.sh ./ccminer --benchmark
