## Frfun7 ##

Frfun7 is a spatial fractal denoising plugin by 
Copyright (C) 2002-2006, 2013 by Marc Fauconneau (prunedtree), (C)2021 Ferenc Pintér

### Usage
```
frfun7 (clip, float "lambda", float "T", float "Tuv", int "P", int "TP", int "R1")
```
         clip   =
    
            Input clip. 


        float  lambda = 1.1
    
            Adjust the power of the local denoising.


        float  T = 6.0
    
            Limits the max luma denoising power for edges; 0 disables processing. 


        float  Tuv = 2.0
    
            Limits the max chroma denoising power for edges; 0 disables processing. 


        int  P = 0
    
            By testing, one can conclude it's a "speed -vs- quality" trade off setting. 
    
                    0 : faster but slightly lower quality than frfun7_rev6 (may create minor artifacts around line edges).
                    1 : adaptive overlapping (see also TP1). slower than frfun7_rev6 but the quality is a little bit better.
                    2 : temporal
                    4 : adaptive radius
    
            Internally the parameter is treated as a bit mask but probably it has no point.
    
            Parameter is available since frfun7 2013.


        int  TP1 = 0
    
            A threshold which affects P=1 (adaptive overlapping).
            Introduced as a separate parameter in r0.7test. This value had to be encoded into P as TP1*1000 previously.
            0 will always run into a final filtering part, the bigger it is, probably the more pixels it will skip. (?)

        int  R1 = 3
    
            Radius for first pass of the internal algorithm.
            First pass in pre v0.7 was fixed to 3 (and was no separate parameter)
            Valid values are 2 or 3.

  frfun7 with default settings:

  ```
  AviSource("Blah.avi")
  frfun7(lambda=1.1, T=6.0, Tuv=2.0, P=0, TP1=0, R1=3)
  ```

### Known issues

P=2 (temporal) has sometimes blocky rectangular artifacts at the most top and bottom area

### Links

http://avisynth.nl/index.php/Frfun7

### Other

Special thanks to Reel.Deel for the testing and comparison with previous versions and maintaining the documentation at avisynth.nl.


Build instructions
==================
## Visual Studio 2019: 

use IDE

## Windows GCC

(mingw installed by msys2)
From the 'build' folder under project root:

```
del ..\CMakeCache.txt
cmake .. -G "MinGW Makefiles" -DENABLE_INTEL_SIMD:bool=on
cmake --build . --config Release 
```

## Linux

from the 'build' folder under project root:
ENABLE_INTEL_SIMD is automatically off for non-x86 architectures
Note: plugin source only supports INTEL

* Clone repo and build
  
        git clone https://github.com/pinterf/Frfun7
        cd Frfun7
        cmake -B build -S .
        cmake --build build

  Useful hints:        
   build after clean:

      cmake --build build --clean-first

   Force no assembler support (presently it is not valid; plugin relies on Intel assembler code inside)
  
    cmake -B build -S . -DENABLE_INTEL_SIMD:bool=off
  

 delete CMake cache

    rm build/CMakeCache.txt

* Find binaries at
  
        build/Frfun7/frfun7.so

* Install binaries

        cd build
        sudo make install

### History
```
Version         Date            Changes
0.7             2021/09/09      - release

0.7 WIP         2021/05/25      - re-enable T=0, Tuv=0 cases (unprocessed plane copy)
                                - add experimental TP1 (default 0) a threshold for P=1 (temporal overlapping) mode
                                - add experimental R1 (default 3, can be set to 2) first pass radius

0.7 WIP         2021/05/20      - Source is based on a 2006/05/11 snapshot 
                                - Code refresh and additions by pinterf
                                - move to git: https://github.com/pinterf/Frfun7
                                - Source to VS2019 solution
                                - Add readme, usage based on Avisynth wiki
                                - Add/guess missing source parts, rename files
                                - Update AviSynth headers
                                - Add version resource
                                - Avisynth V2.6 style plugin
                                - Implement all mmx inline assembler as SIMD intrinsics
                                - x64 build
                                - fix some rounding and other issue

2013            2013/09/04      - no longer buffers the input; yields a nice speed increase.
                                - "P" parameter added
                                  Note: frfun7 2013 is the updated version, unfortunately the output is 
                                  not completely identical to frfun7_rev6, for that reason both versions
                                  are available. Read description of the P parameter for more information. 

rev6            2006/05/10      - bug fixes
                                - remove mod8 restriction
                                - process first and last frame

rev1            2006/05/05      - initial release. 

```

