//
//  Frfun7 - Avisynth filter 
//  Copyright (C) 2006, 2013 Marc Fauconneau
//            (C) 2021 Ferenc Pintér
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

#ifndef __FRFUN7_H__
#define __FRFUN7_H__

#include "avisynth.h"
#include "stdint.h"

class AvsFilter : public GenericVideoFilter
{
  bool has_at_least_v8;

  int inv_table[1024];
  int lambda, Thresh_luma, Thresh_chroma;
  int P;
  int P1_param;
  int R_1stpass; // Radius of first pass, originally 3, can be 2 as well
  int opt;

  uint8_t* wpln; // weight buffer videosize_x/4,videosize_y/4
  int wp_stride;
  int wp_width;
  int wp_height;
public:
  AvsFilter(AVSValue args, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env);
  ~AvsFilter();
};

#endif /* __FRFUN7_H__ */
