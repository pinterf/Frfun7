## Frfun7 ##

Frfun7 is a spatial fractal denoising plugin by 
Copyright (C) 2002-2006, 2013 by Marc Fauconneau (prunedtree), (C)2021 Ferenc Pintér

### Usage
```
frfun7 (clip, float "lambda",float "T", float "Tuv", int "P")
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
    
            Undocumented parameter, only available in frfun7 2013. By testing, one can conclude it's a "speed -vs- quality" trade off setting. 
    
                    0 : faster but slightly lower quality than frfun7_rev6 (may create minor artifacts around line edges).
                    1 : slower than frfun7_rev6 but the quality is a little bit better. 
    
            2021 finding by pinterf: 
            P is bit mask, plus a *1000 value for P and 1
                    P lsb 3 bits are zero: ???
                    P and 1: adaptive overlapping
                    P and 2: temporal
                    P and 4: adaptive radius
            When P and 1, then P/1000 is defining an additional threshold on internal weight table, e.g. P = 12*1000 + 1
            * 20210526, this /1000-parameter available as a separate parameter TP1
            Source dated on 2006/05/11 already contains parameter P. Probably it was disabled for rev6 release (6 days before).

        int  TP1 = 0
    
            A threshold which affects P=1 (adaptive overlapping).
            Introduced as a separate parameter in r0.7test. This value had to be encoded into P as TP1*1000 previously.
            0 will always run into a final filtering part, the bigger it is, probably the more pixels it will skip. (?)
               

        int  R1 = 3
    
            Introduced in r0.7test, experimental. Algorith'm first pass was fixed to R1=3, now it can be set to R1=2.

  frfun7 with default settings:

  ```
  AviSource("Blah.avi")
  frfun7(lambda=1.1, T=6.0, Tuv=2.0, P=0, TP1=0, R1=3)
  ```

### Known issues

P=2 (temporal) has sometimes blocky rectangular artifacts at the most top and bottom area
Possibly is was experimental

### Links

http://avisynth.nl/index.php/Frfun7

### History
```
Version         Date            Changes
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

