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
#include "avs/alignment.h"
#include "stdint.h"

struct Vect {
  int x, y;
};

Vect GetAVSDim(PVideoFrame f, int plane)
{
  Vect dim;
  dim.x = f->GetRowSize(plane);
  dim.y = f->GetHeight(plane);
  //stride = f->GetPitch(plane);
  return dim;
}

int clipb(int weight) {
  return weight < 0 ? 0 : weight > 255 ? 255 : weight;
}

class Plane8i {
  unsigned char* buf;
  int origin_x, origin_y;
private:
  void update_ptr()
  {
    if (buf)
      ptr = const_cast<unsigned char*>(buf + stride * origin_y + origin_x);
    else
      ptr = nullptr;
  }
public:
  int w;
  int h;
  int stride;
  unsigned char* ptr;

  unsigned char* get_ptr(int x, int y) const { return buf + stride * (origin_y + y) + origin_x + x; }

  void set_buf(unsigned char* _buf) {
    buf = _buf;
    update_ptr();
  }

  Plane8i() : buf(nullptr), origin_x(0), origin_y(0), w(0), h(0), stride(0) {
    update_ptr();
  };

  void free() {
    if (buf) {
      _aligned_free(buf);
      buf = nullptr;
    }
  }

  Plane8i alloc(int w, int h) {
    Plane8i newplanebuf = *this;
    newplanebuf.w = w;
    newplanebuf.h = h;
    const int ALIGN = 32;
    int aligned_w = (((w)+(ALIGN)-1) & (~((ALIGN)-1)));
    newplanebuf.stride = aligned_w;
    newplanebuf.set_buf((unsigned char*)_aligned_malloc(stride * h, ALIGN));
    return newplanebuf;
  }


  Plane8i(PVideoFrame* f, int plane, bool read) : origin_x(0), origin_y(0) {
    // PVideoFrame* because of possible GetWritePtr
    w = (*f)->GetRowSize(plane);
    h = (*f)->GetHeight(plane);
    stride = (*f)->GetPitch(plane);
    set_buf(read ? const_cast<unsigned char*>((*f)->GetReadPtr(plane)) : (*f)->GetWritePtr(plane));
    update_ptr();
  };

  Plane8i& operator=(const Plane8i& p) {
    buf = p.buf;
    origin_x = p.origin_x;
    origin_y = p.origin_y;
    w = p.w;
    h = p.h;
    stride = p.stride;
    ptr = p.ptr;
    return *this;
  }

  const Plane8i operator()(int x, int y) {
    Plane8i shifted = *this;
    shifted.origin_x += x;
    shifted.origin_y += y;
    shifted.update_ptr();
    return shifted;
  }
};

Plane8i ImportAVSRead(PVideoFrame* f, int plane)
{
  return Plane8i(f, plane, true);
}

Plane8i ImportAVSWrite(PVideoFrame* f, int plane)
{
  return Plane8i(f, plane, false);
}

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
