//
//	PostProc (version x.x) - Avisynth filter 
//	Copyright (C) 2002 Marc Fauconneau
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
//
//  Please contact me for any bugs or questions.
//  marc.fd@libertysurf.fr

//  Change log :
//         20/07/2002 - ver x.x  - Avisynth filter coded (from scratch)

#ifndef __FRFUN7_H__
#define __FRFUN7_H__

#include "avisynth.h"
#include "..\(image lib)\image.h"

#define uint8_t unsigned char

class AvsFilter : public GenericVideoFilter
{
  int lastn;
  PVideoFrame pf, cf;

  int inv_table[1024];
  int lambda, T, Tuv;
  int P;
  Plane8i wpln;
public:
  AvsFilter(AVSValue args, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env);
  ~AvsFilter();
};

#endif /* __FRFUN7_H__ */
