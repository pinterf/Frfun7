
#include "avisynth.h"
#include "Frfun7.h"

#include <math.h>

// trash 2,3 result in 2
#define sad16(offset, rdst, rdp, rsad, rtmp)		\
__asm				movd	mm2, [rdst+offset]		\
__asm				movd	mm3, [rdst+rdp+offset]	\
__asm				psllq	mm2, 32					\
__asm				por		mm2, mm3				\
__asm				psadbw	mm2, mm0				\
__asm				movd	rtmp, mm2					\
__asm				movd	mm2, [rdst+rdp*2+offset]	\
__asm				add		rdst, rdp					\
__asm				movd	mm3, [rdst+rdp*2+offset]	\
__asm				sub		rdst, rdp					\
__asm				psllq	mm2, 32					\
__asm				por		mm2, mm3				\
__asm				psadbw	mm2, mm1				\
__asm				movd	rsad, mm2				\
__asm				add		rsad, rtmp

// trash rsad,rb
#define comp(rsad,rtmp,racc,rthr)				\
__asm				cmp		rsad, rthr			\
__asm				mov		rsad, 0				\
__asm				mov		rtmp, 0xFFFFFFFF	\
__asm				cmovl	rsad, rtmp			\
__asm				and		rsad, 1				\
__asm				add		racc,rsad			\
__asm				shl		rsad, 31			\
__asm				sar		rsad, 31

// trash rsad,rb
#define ____comp(rsad,rtmp,racc,rthr)				\
__asm				cmp		rsad, rthr			\
__asm				mov		rsad, 0				\
__asm				mov		rtmp, 1				\
__asm				cmovl	rsad, rtmp			\
__asm				add		racc, rsad			\
__asm				shl		rsad, 31			\
__asm				sar		rsad, 31			\
__asm				movd	mm2, rsad

#define	acc4(mmA,rsad)						\
__asm				movd		mm2, rsad	\
__asm				pand		mm3, mm2 	\
__asm				pxor		mm2, mm2 	\
__asm				punpcklbw	mm3, mm2	\
__asm				paddw		mmA, mm3

#define	acc16(offset, rdst, rdp, rsad)					\
__asm				movd	mm3, [rdst+offset]			\
__asm				acc4(	mm4, rsad)					\
__asm				movd	mm3, [rdst+rdp+offset]		\
__asm				acc4(	mm5, rsad)					\
__asm				movd	mm3, [rdst+rdp*2+offset]	\
__asm				acc4(	mm6, rsad)					\
__asm				add		rdst, rdp					\
__asm				movd	mm3, [rdst+rdp*2+offset]	\
__asm				acc4(	mm7, rsad)					\
__asm				sub		rdst, rdp

#define check(offset, rdst, rdp, rsad, rtmp, racc, rthr)	\
__asm				sad16(offset, rdst, rdp, rsad, rtmp)	\
__asm				comp(rsad, rtmp, racc, rthr)			\
__asm				acc16(offset, rdst, rdp, rsad)

#define acheck(offset, rdst, rdp, rsad, rtmp, racc, rthr)	\
__asm				sad16(offset, rdst, rdp, rsad, rtmp)	\
__asm				add edx, rsad							\
__asm				comp(rsad, rtmp, racc, rthr)			\
__asm				acc16(offset, rdst, rdp, rsad)

/*
#define	___stor4(mmA)						\
__asm				movq mm1, mmA			\
__asm				punpcklwd mm1,mm0		\
__asm				movq mm2, [esi]			\
__asm				paddd mm2, mm1			\
__asm				movq [esi], mm2			\
__asm				movq mm1, mmA			\
__asm				punpckhwd mm1,mm0		\
__asm				movq mm2, [esi+8]		\
__asm				paddd mm2, mm1			\
__asm				movq [esi+8], mm2
*/
#define	stor4(mmA)								\
__asm				movq	mm1, mmA			\
__asm				psllw	mm1, 2				\
__asm				pmulhuw	mm1,mm0				\
__asm				paddusw	mm1,mm3				\
__asm				psrlw	mm1,1				\
__asm				packuswb mm1,mm2			\
__asm				movd	[esi],mm1

void frcore_filter_b4r3_mmx(Plane8i rpln, Plane8i cpln, Plane8i dpln, int T, int* inv_table, int* weight)
{

  uint8_t* ptrr = rpln.ptr;
  uint8_t* ptra = cpln(-3, -3).ptr;
  uint8_t* ptrb = dpln.ptr;

  int* _weight = weight;
  int* _inv_table = inv_table;
  int _thresh = T;
  int pitchr = rpln.stride;
  int pitcha = cpln.stride;
  int pitchb = dpln.stride;

  __asm
  {
    push ebx; safety^^

    mov edx, [_thresh]
    xor ecx, ecx

    mov esi, ptrr
    mov eax, pitchr

    movd mm0, [esi]
    movd mm1, [esi + eax]
    movd mm2, [esi + eax * 2]
    add esi, eax
    movd mm3, [esi + eax * 2]

    psllq	mm0, 32
    por		mm0, mm1
    psllq	mm2, 32
    por		mm2, mm3
    movq	mm1, mm2

    mov esi, ptra
    mov eax, pitcha

    pxor mm4, mm4
    pxor mm5, mm5
    pxor mm6, mm6
    pxor mm7, mm7

    ; -3
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)
    check(5, esi, eax, edi, ebx, ecx, edx)
    check(6, esi, eax, edi, ebx, ecx, edx)

    add esi, eax
    ; -2
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)
    check(5, esi, eax, edi, ebx, ecx, edx)
    check(6, esi, eax, edi, ebx, ecx, edx)

    add esi, eax
    ; -1
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)
    check(5, esi, eax, edi, ebx, ecx, edx)
    check(6, esi, eax, edi, ebx, ecx, edx)

    add esi, eax
    ; 0
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)
    check(5, esi, eax, edi, ebx, ecx, edx)
    check(6, esi, eax, edi, ebx, ecx, edx)

    add esi, eax
    ; +1
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)
    check(5, esi, eax, edi, ebx, ecx, edx)
    check(6, esi, eax, edi, ebx, ecx, edx)
    add esi, eax

    ; +2
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)
    check(5, esi, eax, edi, ebx, ecx, edx)
    check(6, esi, eax, edi, ebx, ecx, edx)
    add esi, eax

    ; +3
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)
    check(5, esi, eax, edi, ebx, ecx, edx)
    check(6, esi, eax, edi, ebx, ecx, edx)

    ; 0 - 3 can be thrashed from here. 4 - 7 has acc, ecx has weight

    mov		eax, _weight
    mov[eax], ecx

    mov esi, ptrb
    mov eax, pitchb
    pxor mm2, mm2
    pcmpeqd mm3, mm3
    psrlw	mm3, 15

    ; scale 4 - 7 by ecx

    mov		ebx, _inv_table
    movd	mm0, [ebx + ecx * 4]
    movq	mm1, mm0
    psllq	mm1, 32
    por		mm0, mm1
    movq	mm1, mm0
    psllq	mm1, 16
    por		mm0, mm1

    stor4(mm4)
    add		esi, eax

    stor4(mm5)
    add		esi, eax

    stor4(mm6)
    add		esi, eax

    stor4(mm7)

    pop ebx; safety^^
  }
}

void frcore_filter_adapt_b4r3_mmx(Plane8i rpln, Plane8i cpln, Plane8i dpln, int T, int sT2, int sT3, int* inv_table, int* weight)
{

  uint8_t* ptrr = rpln.ptr;
  uint8_t* ptra = cpln(-3, -1).ptr;
  uint8_t* ptrb = dpln.ptr;

  int* _weight = weight;
  int* _inv_table = inv_table;
  int _thresh = T;
  int _thresh2 = sT2;
  int _thresh3 = sT3;
  int pitchr = rpln.stride;
  int pitcha = cpln.stride;
  int pitchb = dpln.stride;

  __asm
  {
    push ebx; safety^^

    //	mov edx, [_thresh]
    xor ecx, ecx

    mov esi, ptrr
    mov eax, pitchr

    movd mm0, [esi]
    movd mm1, [esi + eax]
    movd mm2, [esi + eax * 2]
    add esi, eax
    movd mm3, [esi + eax * 2]

    psllq	mm0, 32
    por		mm0, mm1
    psllq	mm2, 32
    por		mm2, mm3
    movq	mm1, mm2

    mov esi, ptra
    mov eax, pitcha

    pxor mm4, mm4
    pxor mm5, mm5
    pxor mm6, mm6
    pxor mm7, mm7
    xor edx, edx

    ; -1
    acheck(2, esi, eax, edi, ebx, ecx, _thresh)
    acheck(3, esi, eax, edi, ebx, ecx, _thresh)
    acheck(4, esi, eax, edi, ebx, ecx, _thresh)

    add esi, eax
    ; 0
    acheck(2, esi, eax, edi, ebx, ecx, _thresh)
    acheck(3, esi, eax, edi, ebx, ecx, _thresh)
    acheck(4, esi, eax, edi, ebx, ecx, _thresh)

    add esi, eax
    ; +1
    acheck(2, esi, eax, edi, ebx, ecx, _thresh)
    acheck(3, esi, eax, edi, ebx, ecx, _thresh)
    acheck(4, esi, eax, edi, ebx, ecx, _thresh)

    cmp edx, _thresh2
    jl __frcore_filter_adapt_b4r3_mmx___finish

    sub esi, eax; 0
    sub esi, eax; -1
    sub esi, eax
    ; -2
    acheck(1, esi, eax, edi, ebx, ecx, _thresh)
    acheck(2, esi, eax, edi, ebx, ecx, _thresh)
    acheck(3, esi, eax, edi, ebx, ecx, _thresh)
    acheck(4, esi, eax, edi, ebx, ecx, _thresh)
    acheck(5, esi, eax, edi, ebx, ecx, _thresh)

    add esi, eax
    ; -1
    acheck(1, esi, eax, edi, ebx, ecx, _thresh)
    acheck(5, esi, eax, edi, ebx, ecx, _thresh)

    add esi, eax
    ; 0
    acheck(1, esi, eax, edi, ebx, ecx, _thresh)
    acheck(5, esi, eax, edi, ebx, ecx, _thresh)

    add esi, eax
    ; +1
    acheck(1, esi, eax, edi, ebx, ecx, _thresh)
    acheck(5, esi, eax, edi, ebx, ecx, _thresh)

    add esi, eax
    ; +2
    acheck(1, esi, eax, edi, ebx, ecx, _thresh)
    acheck(2, esi, eax, edi, ebx, ecx, _thresh)
    acheck(3, esi, eax, edi, ebx, ecx, _thresh)
    acheck(4, esi, eax, edi, ebx, ecx, _thresh)
    acheck(5, esi, eax, edi, ebx, ecx, _thresh)

    cmp edx, _thresh3
    jl __frcore_filter_adapt_b4r3_mmx___finish

    sub esi, eax; +1
    sub esi, eax; 0
    sub esi, eax; -1
    sub esi, eax; -2
    sub esi, eax
    ; -3
    acheck(0, esi, eax, edi, ebx, ecx, _thresh)
    acheck(1, esi, eax, edi, ebx, ecx, _thresh)
    acheck(2, esi, eax, edi, ebx, ecx, _thresh)
    acheck(3, esi, eax, edi, ebx, ecx, _thresh)
    acheck(4, esi, eax, edi, ebx, ecx, _thresh)
    acheck(5, esi, eax, edi, ebx, ecx, _thresh)
    acheck(6, esi, eax, edi, ebx, ecx, _thresh)

    add esi, eax
    ; -2
    acheck(0, esi, eax, edi, ebx, ecx, _thresh)
    acheck(6, esi, eax, edi, ebx, ecx, _thresh)

    add esi, eax
    ; -1
    acheck(0, esi, eax, edi, ebx, ecx, _thresh)
    acheck(6, esi, eax, edi, ebx, ecx, _thresh)

    add esi, eax
    ; 0
    acheck(0, esi, eax, edi, ebx, ecx, _thresh)
    acheck(6, esi, eax, edi, ebx, ecx, _thresh)

    add esi, eax
    ; +1
    acheck(0, esi, eax, edi, ebx, ecx, _thresh)
    acheck(6, esi, eax, edi, ebx, ecx, _thresh)

    add esi, eax
    ; +2
    acheck(0, esi, eax, edi, ebx, ecx, _thresh)
    acheck(6, esi, eax, edi, ebx, ecx, _thresh)

    add esi, eax
    ; -3
    acheck(0, esi, eax, edi, ebx, ecx, _thresh)
    acheck(1, esi, eax, edi, ebx, ecx, _thresh)
    acheck(2, esi, eax, edi, ebx, ecx, _thresh)
    acheck(3, esi, eax, edi, ebx, ecx, _thresh)
    acheck(4, esi, eax, edi, ebx, ecx, _thresh)
    acheck(5, esi, eax, edi, ebx, ecx, _thresh)
    acheck(6, esi, eax, edi, ebx, ecx, _thresh)


    __frcore_filter_adapt_b4r3_mmx___finish :

    ; 0 - 3 can be thrashed from here. 4 - 7 has acc, ecx has weight

      mov		eax, _weight
      mov[eax], ecx

      mov esi, ptrb
      mov eax, pitchb
      pxor mm2, mm2
      pcmpeqd mm3, mm3
      psrlw	mm3, 15

      ; scale 4 - 7 by ecx

      mov		ebx, _inv_table
      movd	mm0, [ebx + ecx * 4]
      movq	mm1, mm0
      psllq	mm1, 32
      por		mm0, mm1
      movq	mm1, mm0
      psllq	mm1, 16
      por		mm0, mm1

      stor4(mm4)
      add		esi, eax

      stor4(mm5)
      add		esi, eax

      stor4(mm6)
      add		esi, eax

      stor4(mm7)

      pop ebx; safety^^
  }
}

void frcore_filter_b4r0_mmx(Plane8i rpln, Plane8i cpln, Plane8i dpln, int T, int* inv_table, int* weight)
{

  uint8_t* ptrr = rpln.ptr;
  uint8_t* ptra = cpln(-0, -0).ptr;
  uint8_t* ptrb = dpln.ptr;

  int* _weight = weight;
  int* _inv_table = inv_table;
  int _thresh = T;
  int pitchr = rpln.stride;
  int pitcha = cpln.stride;
  int pitchb = dpln.stride;

  __asm
  {
    push ebx; safety^^

    mov edx, [_thresh]
    xor ecx, ecx

    mov esi, ptrr
    mov eax, pitchr

    movd mm0, [esi]
    movd mm1, [esi + eax]
    movd mm2, [esi + eax * 2]
    add esi, eax
    movd mm3, [esi + eax * 2]

    psllq	mm0, 32
    por		mm0, mm1
    psllq	mm2, 32
    por		mm2, mm3
    movq	mm1, mm2

    mov esi, ptra
    mov eax, pitcha

    pxor mm4, mm4
    pxor mm5, mm5
    pxor mm6, mm6
    pxor mm7, mm7

    ; 0
    check(0, esi, eax, edi, ebx, ecx, edx)

    ; 0 - 3 can be thrashed from here. 4 - 7 has acc, ecx has weight

    mov		eax, _weight
    mov[eax], ecx

    mov esi, ptrb
    mov eax, pitchb
    pxor mm2, mm2
    pcmpeqd mm3, mm3
    psrlw	mm3, 15

    ; scale 4 - 7 by ecx

    mov		ebx, _inv_table
    movd	mm0, [ebx + ecx * 4]
    movq	mm1, mm0
    psllq	mm1, 32
    por		mm0, mm1
    movq	mm1, mm0
    psllq	mm1, 16
    por		mm0, mm1

    stor4(mm4)
    add		esi, eax

    stor4(mm5)
    add		esi, eax

    stor4(mm6)
    add		esi, eax

    stor4(mm7)

    pop ebx; safety^^
  }
}

#define	blend_store4(mmA)						\
__asm			movd	mm3, [esi]				\
__asm			punpcklbw mm3, mm0				\
__asm			psllw	mm3, 6					\
__asm			pmulhw	mm3, mm2				\
__asm			paddusw	mmA, mm3				\
__asm			paddusw	mmA, mm1				\
__asm			psrlw	mmA, 5					\
__asm			packuswb mmA, mm0				\
__asm			movd	[esi], mmA

#define	__blend_store4(mmA)						\
__asm			movd	mm3, [esi]				\
__asm			punpcklbw mm3, mm0				\
__asm			paddw	mmA, mm3				\
__asm			paddw	mmA, mm1				\
__asm			psrlw	mmA, 1					\
__asm			packuswb mmA, mm0				\
__asm			movd	[esi], mmA


#define expand_word(mmA,mmB)				\
__asm			movq	mmB, mmA			\
__asm			psllq	mmA, 32				\
__asm			por		mmA, mmB			\
__asm			movq	mmB, mmA			\
__asm			psllq	mmB, 16				\
__asm			por		mmA, mmB


void frcore_filter_overlap_b4r3_mmx(Plane8i rpln, Plane8i cpln, Plane8i dpln, int T, int* inv_table, int* weight)
{

  uint8_t* ptrr = rpln.ptr;
  uint8_t* ptra = cpln(-3, -3).ptr;
  uint8_t* ptrb = dpln.ptr;

  int* _weight = weight;
  int* _inv_table = inv_table;
  int _thresh = T;
  int pitchr = rpln.stride;
  int pitcha = cpln.stride;
  int pitchb = dpln.stride;

  __asm
  {
    push ebx; safety^^

    mov edx, [_thresh]
    xor ecx, ecx

    mov esi, ptrr
    mov eax, pitchr

    movd mm0, [esi]
    movd mm1, [esi + eax]
    movd mm2, [esi + eax * 2]
    add esi, eax
    movd mm3, [esi + eax * 2]

    psllq	mm0, 32
    por		mm0, mm1
    psllq	mm2, 32
    por		mm2, mm3
    movq	mm1, mm2

    mov esi, ptra
    mov eax, pitcha

    pxor mm4, mm4
    pxor mm5, mm5
    pxor mm6, mm6
    pxor mm7, mm7

    ; -3
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)
    check(5, esi, eax, edi, ebx, ecx, edx)
    check(6, esi, eax, edi, ebx, ecx, edx)

    add esi, eax
    ; -2
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)
    check(5, esi, eax, edi, ebx, ecx, edx)
    check(6, esi, eax, edi, ebx, ecx, edx)

    add esi, eax
    ; -1
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)
    check(5, esi, eax, edi, ebx, ecx, edx)
    check(6, esi, eax, edi, ebx, ecx, edx)

    add esi, eax
    ; 0
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)
    check(5, esi, eax, edi, ebx, ecx, edx)
    check(6, esi, eax, edi, ebx, ecx, edx)

    add esi, eax
    ; +1
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)
    check(5, esi, eax, edi, ebx, ecx, edx)
    check(6, esi, eax, edi, ebx, ecx, edx)

    add esi, eax
    ; +2
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)
    check(5, esi, eax, edi, ebx, ecx, edx)
    check(6, esi, eax, edi, ebx, ecx, edx)

    add esi, eax
    ; +3
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)
    check(5, esi, eax, edi, ebx, ecx, edx)
    check(6, esi, eax, edi, ebx, ecx, edx)

    ; 0 - 3 can be thrashed from here. 4 - 7 has acc, ecx has weight

    mov		eax, _weight
    mov		edx, [eax]
    mov[eax], ecx

    mov esi, ptrb
    mov eax, pitchb
    pxor mm2, mm2

    ; scale 4 - 7 by ecx and store(here with blending)

    mov		ebx, _inv_table
    movd	mm1, [ebx + ecx * 4]
    expand_word(mm1, mm0)

    pmulhw	mm4, mm1
    pmulhw	mm5, mm1
    pmulhw	mm6, mm1
    pmulhw	mm7, mm1

    mov		ebx, edx
    and edx, 0xFFFF
    movd	mm2, edx
    expand_word(mm2, mm0)

    psllq	mm4, 7
    psllq	mm5, 7
    psllq	mm6, 7
    psllq	mm7, 7

    pmulhw	mm4, mm2
    pmulhw	mm5, mm2
    pmulhw	mm6, mm2
    pmulhw	mm7, mm2

    shr		ebx, 16
    movd	mm2, ebx
    expand_word(mm2, mm0)

    pxor mm0, mm0
    pcmpeqd mm1, mm1
    psrlw	mm1, 14
    psllw	mm1, 3

    blend_store4(mm4)

    add		esi, eax
    blend_store4(mm5)

    add		esi, eax
    blend_store4(mm6)

    add		esi, eax
    blend_store4(mm7)

    pop ebx; safety^^
  }
}

void frcore_filter_overlap_b4r2_mmx(Plane8i rpln, Plane8i cpln, Plane8i dpln, int T, int* inv_table, int* weight)
{

  uint8_t* ptrr = rpln.ptr;
  uint8_t* ptra = cpln(-2, -2).ptr;
  uint8_t* ptrb = dpln.ptr;

  int* _weight = weight;
  int* _inv_table = inv_table;
  int _thresh = T;
  int pitchr = rpln.stride;
  int pitcha = cpln.stride;
  int pitchb = dpln.stride;

  __asm
  {
    push ebx; safety^^

    mov edx, [_thresh]
    xor ecx, ecx

    mov esi, ptrr
    mov eax, pitchr

    movd mm0, [esi]
    movd mm1, [esi + eax]
    movd mm2, [esi + eax * 2]
    add esi, eax
    movd mm3, [esi + eax * 2]

    psllq	mm0, 32
    por		mm0, mm1
    psllq	mm2, 32
    por		mm2, mm3
    movq	mm1, mm2

    mov esi, ptra
    mov eax, pitcha

    pxor mm4, mm4
    pxor mm5, mm5
    pxor mm6, mm6
    pxor mm7, mm7

    ; -2
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)

    add esi, eax
    ; -1
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)

    add esi, eax
    ; 0
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)

    add esi, eax
    ; +1
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)

    add esi, eax
    ; +2
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)

    ; 0 - 3 can be thrashed from here. 4 - 7 has acc, ecx has weight

    mov		eax, _weight
    mov		edx, [eax]
    mov[eax], ecx

    mov esi, ptrb
    mov eax, pitchb
    pxor mm2, mm2

    ; scale 4 - 7 by ecx and store(here with blending)

    mov		ebx, _inv_table
    movd	mm1, [ebx + ecx * 4]
    expand_word(mm1, mm0)

    pmulhw	mm4, mm1
    pmulhw	mm5, mm1
    pmulhw	mm6, mm1
    pmulhw	mm7, mm1

    mov		ebx, edx
    and edx, 0xFFFF
    movd	mm2, edx
    expand_word(mm2, mm0)

    psllq	mm4, 7
    psllq	mm5, 7
    psllq	mm6, 7
    psllq	mm7, 7

    pmulhw	mm4, mm2
    pmulhw	mm5, mm2
    pmulhw	mm6, mm2
    pmulhw	mm7, mm2

    shr		ebx, 16
    movd	mm2, ebx
    expand_word(mm2, mm0)

    pxor mm0, mm0
    pcmpeqd mm1, mm1
    psrlw	mm1, 14
    psllw	mm1, 3

    blend_store4(mm4)

    add		esi, eax
    blend_store4(mm5)

    add		esi, eax
    blend_store4(mm6)

    add		esi, eax
    blend_store4(mm7)

    pop ebx; safety^^
  }
}

void frcore_filter_overlap_b4r0_mmx(Plane8i rpln, Plane8i cpln, Plane8i dpln, int T, int* inv_table, int* weight)
{

  uint8_t* ptrr = rpln.ptr;
  uint8_t* ptra = cpln(0, 0).ptr;
  uint8_t* ptrb = dpln.ptr;

  int* _weight = weight;
  int* _inv_table = inv_table;
  int _thresh = T;
  int pitchr = rpln.stride;
  int pitcha = cpln.stride;
  int pitchb = dpln.stride;

  __asm
  {
    push ebx; safety^^

    mov edx, [_thresh]
    xor ecx, ecx

    mov esi, ptrr
    mov eax, pitchr

    movd mm0, [esi]
    movd mm1, [esi + eax]
    movd mm2, [esi + eax * 2]
    add esi, eax
    movd mm3, [esi + eax * 2]

    psllq	mm0, 32
    por		mm0, mm1
    psllq	mm2, 32
    por		mm2, mm3
    movq	mm1, mm2

    mov esi, ptra
    mov eax, pitcha

    pxor mm4, mm4
    pxor mm5, mm5
    pxor mm6, mm6
    pxor mm7, mm7

    ; 0
    check(0, esi, eax, edi, ebx, ecx, edx)

    ; 0 - 3 can be thrashed from here. 4 - 7 has acc, ecx has weight

    mov		eax, _weight
    mov		edx, [eax]
    mov[eax], ecx

    mov esi, ptrb
    mov eax, pitchb
    pxor mm2, mm2

    ; scale 4 - 7 by ecx and store(here with blending)

    mov		ebx, _inv_table
    movd	mm1, [ebx + ecx * 4]
    expand_word(mm1, mm0)

    /*	pmulhw	mm4, mm1
      pmulhw	mm5, mm1
      pmulhw	mm6, mm1
      pmulhw	mm7, mm1 */

      mov		ebx, edx
      and edx, 0xFFFF
      movd	mm2, edx
      expand_word(mm2, mm0)

      psllq	mm4, 6
      psllq	mm5, 6
      psllq	mm6, 6
      psllq	mm7, 6

      pmulhw	mm4, mm2
      pmulhw	mm5, mm2
      pmulhw	mm6, mm2
      pmulhw	mm7, mm2

      shr		ebx, 16
      movd	mm2, ebx
      expand_word(mm2, mm0)

      pxor mm0, mm0
      pcmpeqd mm1, mm1
      psrlw	mm1, 14
      psllw	mm1, 3

      blend_store4(mm4)

      add		esi, eax
      blend_store4(mm5)

      add		esi, eax
      blend_store4(mm6)

      add		esi, eax
      blend_store4(mm7)

      pop ebx; safety^^
  }
}

#define	blend_diff4(mmA)						\
__asm			movd	mm3, [esi]				\
__asm			punpcklbw mm3, mm0				\
__asm			psllw	mm3, 6					\
__asm			pmulhw	mm3, mm2				\
__asm			paddw	mmA, mm3				\
__asm			paddw	mmA, mm1				\
__asm			psrlw	mmA, 5					\
__asm			packuswb mmA, mm0				\
__asm			movd	mm3, [esi]				\
__asm			psadbw	mmA, mm3


void frcore_filter_diff_b4r1_mmx(Plane8i rpln, Plane8i cpln, Plane8i dpln, int T, int* inv_table, int* weight)
{

  uint8_t* ptrr = rpln.ptr;
  uint8_t* ptra = cpln(-1, -1).ptr;
  uint8_t* ptrb = dpln.ptr;

  int* _weight = weight;
  int* _inv_table = inv_table;
  int _thresh = T;
  int pitchr = rpln.stride;
  int pitcha = cpln.stride;
  int pitchb = dpln.stride;

  __asm
  {
    push ebx; safety^^

    mov edx, [_thresh]
    xor ecx, ecx

    mov esi, ptrr
    mov eax, pitchr

    movd mm0, [esi]
    movd mm1, [esi + eax]
    movd mm2, [esi + eax * 2]
    add esi, eax
    movd mm3, [esi + eax * 2]

    psllq	mm0, 32
    por		mm0, mm1
    psllq	mm2, 32
    por		mm2, mm3
    movq	mm1, mm2

    mov esi, ptra
    mov eax, pitcha

    pxor mm4, mm4
    pxor mm5, mm5
    pxor mm6, mm6
    pxor mm7, mm7

    ; -1
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)

    add esi, eax
    ; 0
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)

    add esi, eax
    ; +1
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)

    ; 0 - 3 can be thrashed from here. 4 - 7 has acc, ecx has weight

    mov		eax, _weight
    mov		edx, [eax]
    mov[eax], ecx

    mov esi, ptrb
    mov eax, pitchb
    pxor mm2, mm2

    ; scale 4 - 7 by ecx and store(here with blending)

    mov		ebx, _inv_table
    movd	mm1, [ebx + ecx * 4]
    expand_word(mm1, mm0)

    pmulhw	mm4, mm1
    pmulhw	mm5, mm1
    pmulhw	mm6, mm1
    pmulhw	mm7, mm1

    mov		ebx, edx
    and edx, 0xFFFF
    movd	mm2, edx
    expand_word(mm2, mm0)

    psllq	mm4, 7
    psllq	mm5, 7
    psllq	mm6, 7
    psllq	mm7, 7

    pmulhw	mm4, mm2
    pmulhw	mm5, mm2
    pmulhw	mm6, mm2
    pmulhw	mm7, mm2

    shr		ebx, 16
    movd	mm2, ebx
    expand_word(mm2, mm0)

    pxor mm0, mm0
    pcmpeqd mm1, mm1
    psrlw	mm1, 14
    psllw	mm1, 3

    blend_diff4(mm4)

    add		esi, eax
    blend_diff4(mm5)

    add		esi, eax
    blend_diff4(mm6)

    add		esi, eax
    blend_diff4(mm7)

    paddw	mm4, mm5
    paddw	mm6, mm7
    paddw	mm4, mm6
    movd	ecx, mm4
    mov		eax, _weight
    mov[eax], ecx

    pop ebx; safety^^
  }
}

void frcore_filter_b4r2_mmx(Plane8i rpln, Plane8i cpln, Plane8i dpln, int T, int* inv_table)
{

  uint8_t* ptrr = rpln.ptr;
  uint8_t* ptra = cpln(-2, -2).ptr;
  uint8_t* ptrb = dpln.ptr;

  int* _inv_table = inv_table;
  int _thresh = T;
  int pitchr = rpln.stride;
  int pitcha = cpln.stride;
  int pitchb = dpln.stride;

  __asm
  {
    push ebx; safety^^

    mov edx, [_thresh]
    xor ecx, ecx

    mov esi, ptrr
    mov eax, pitchr

    movd mm0, [esi]
    movd mm1, [esi + eax]
    movd mm2, [esi + eax * 2]
    add esi, eax
    movd mm3, [esi + eax * 2]

    psllq	mm0, 32
    por		mm0, mm1
    psllq	mm2, 32
    por		mm2, mm3
    movq	mm1, mm2

    mov esi, ptra
    mov eax, pitcha

    pxor mm4, mm4
    pxor mm5, mm5
    pxor mm6, mm6
    pxor mm7, mm7

    ; -2
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)

    add esi, eax
    ; -1
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)

    add esi, eax
    ; 0
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)

    add esi, eax
    ; +1
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)

    add esi, eax
    ; +2
    check(0, esi, eax, edi, ebx, ecx, edx)
    check(1, esi, eax, edi, ebx, ecx, edx)
    check(2, esi, eax, edi, ebx, ecx, edx)
    check(3, esi, eax, edi, ebx, ecx, edx)
    check(4, esi, eax, edi, ebx, ecx, edx)
    add esi, eax

    ; 0 - 3 can be thrashed from here. 4 - 7 has acc, ecx has weight

    mov esi, ptrb
    mov eax, pitchb
    pxor mm2, mm2

    ; scale 4 - 7 by ecx

    mov		ebx, _inv_table
    movd	mm0, [ebx + ecx * 4]
    movq	mm1, mm0
    psllq	mm1, 32
    por		mm0, mm1
    movq	mm1, mm0
    psllq	mm1, 16
    por		mm0, mm1

    movq	mm1, mm4
    psllw	mm1, 1
    pmulhw	mm1, mm0
    packuswb mm1, mm2
    movd[esi], mm1
    add		esi, eax

    movq	mm1, mm5
    psllw	mm1, 1
    pmulhw	mm1, mm0
    packuswb mm1, mm2
    movd[esi], mm1
    add		esi, eax

    movq	mm1, mm6
    psllw	mm1, 1
    pmulhw	mm1, mm0
    packuswb mm1, mm2
    movd[esi], mm1
    add		esi, eax

    movq	mm1, mm7
    psllw	mm1, 1
    pmulhw	mm1, mm0
    packuswb mm1, mm2
    movd[esi], mm1

    pop ebx; safety^^
  }
}

void frcore_dev_b4_mmx(Plane8i cpln, int* dev)
{

  uint8_t* ptra = cpln(-1, 0).ptr;
  int pitcha = cpln.stride;
  int* _dev = dev;

  __asm
  {
    mov esi, ptra
    mov eax, pitcha

    movd mm0, [esi + 1]
    movd mm1, [esi + eax + 1]
    movd mm2, [esi + eax * 2 + 1]
    add esi, eax
    movd mm3, [esi + eax * 2 + 1]
    sub esi, eax

    psllq	mm0, 32
    por		mm0, mm1
    psllq	mm2, 32
    por		mm2, mm3
    movq	mm1, mm2

    add	esi, eax
    sad16(0, esi, eax, edi, edx)

    sad16(2, esi, eax, ecx, edx)
    cmp		ecx, edi
    cmovl	edi, ecx

    mov		ecx, _dev
    mov[ecx], edi
  }
}

void frcore_sad_b4_mmx(Plane8i apln, Plane8i bpln, int* sad)
{

  uint8_t* ptra = apln(0, 0).ptr;
  int pitcha = apln.stride;
  uint8_t* ptrb = bpln(0, 0).ptr;
  int pitchb = bpln.stride;
  int* _sad = sad;

  __asm
  {
    mov esi, ptra
    mov eax, pitcha

    movd mm0, [esi]
    movd mm1, [esi + eax]
    movd mm2, [esi + eax * 2]
    add esi, eax
    movd mm3, [esi + eax * 2]
    ;	sub esi, eax

    psllq	mm0, 32
    por		mm0, mm1
    psllq	mm2, 32
    por		mm2, mm3
    movq	mm1, mm2

    mov esi, ptrb
    mov eax, pitchb

    sad16(0, esi, eax, edi, edx)

    mov		ecx, _sad
    mov[ecx], edi
  }
}

int get_weight(int alpha)
{
  int a = ((alpha * (1 << 15)) / ((alpha + 1)));
  int b = ((1 << 15) / ((alpha + 1)));
  return (a << 16) | b;
}

PVideoFrame __stdcall AvsFilter::GetFrame(int n, IScriptEnvironment* env)
{
  //	if (n<1 || n>vi.num_frames-1) return child->GetFrame(n, env);
  if (!pf || lastn + 1 != n) pf = child->GetFrame(n - 1, env);
  if (!cf || lastn + 1 != n) cf = child->GetFrame(n, env);
  PVideoFrame nf = child->GetFrame(n + 1, env);
  PVideoFrame df = env->NewVideoFrame(vi);

  int pla = 0, plb = 3;
  /*	if (T==0)
    {
      pla = 1;
      int pl = PLANAR_Y;
      env->BitBlt(df->GetWritePtr(pl),df->GetPitch(pl),cf->GetReadPtr(pl),cf->GetPitch(pl),
        df->GetRowSize(pl),df->GetHeight(pl));
    }
    if (Tuv==0)
    {
      plb = 1;
      int pl = PLANAR_U;
      env->BitBlt(df->GetWritePtr(pl),df->GetPitch(pl),cf->GetReadPtr(pl),cf->GetPitch(pl),
        df->GetRowSize(pl),df->GetHeight(pl));
      pl = PLANAR_V;
      env->BitBlt(df->GetWritePtr(pl),df->GetPitch(pl),cf->GetReadPtr(pl),cf->GetPitch(pl),
        df->GetRowSize(pl),df->GetHeight(pl));
    }
  */

  for (int pl = pla; pl < plb; pl++) {	// PLANES LOOP
    int x, y, plane = (pl == 0) ? (PLANAR_Y) : ((pl == 1) ? (PLANAR_U) : (PLANAR_V));
    Vect dim = GetAVSDim(cf, plane);
    Plane8i ppln = ImportAVSRead(&pf, plane);
    Plane8i cpln = ImportAVSRead(&cf, plane);
    Plane8i npln = ImportAVSRead(&nf, plane);
    Plane8i dpln = ImportAVSWrite(&df, plane);

    int R = 3;
    int B = 4;
    int S = 4;
    int W = R * 2 + 1;

    for (y = 0; y < dim.y + B - 1; y += S)
    {
      int tmax = T;
      if (pl > 0) tmax = Tuv;

      for (x = 0; x < dim.x + B - 1; x += S)
      {
        int sx = x, sy = y;
        int bx = x, by = y;
        if (sx < R) sx = R;
        if (sy < R) sy = R;
        if (sx > dim.x - R - B) sx = dim.x - R - B;
        if (sy > dim.y - R - B) sy = dim.y - R - B;
        if (bx > dim.x - B) bx = dim.x - B;
        if (by > dim.y - B) by = dim.y - B;

        int dev, devp, devn;
        frcore_dev_b4_mmx(cpln(sx, sy), &dev);

        //	P &= ~2;

        if ((P & 2))
        {
          frcore_sad_b4_mmx(cpln(sx, sy), ppln(sx, sy), &devp);
          frcore_sad_b4_mmx(cpln(sx, sy), npln(sx, sy), &devn);
          dev = min(dev, devn);
          dev = min(dev, devp);
        }

        int thresh = ((dev * lambda) >> 10);
        thresh = (thresh > tmax) ? tmax : thresh;
        if (thresh < 1) thresh = 1;

        int weight;
        if ((P & 2))
          frcore_filter_b4r0_mmx(cpln(bx, by), cpln(bx, by), dpln(bx, by), thresh, inv_table, &weight);
        else
        {
          if (sx == x && sy == y && (P & 4))
            frcore_filter_adapt_b4r3_mmx(cpln(bx, by), cpln(sx, sy), dpln(bx, by), thresh, 16 * 9, 16 * 25, inv_table, &weight);
          else
            frcore_filter_b4r3_mmx(cpln(bx, by), cpln(sx, sy), dpln(bx, by), thresh, inv_table, &weight);
        }

        if ((P & 2))
        {
          int k = 1;
          if (devp < thresh)
          {
            weight = get_weight(k);
            frcore_filter_overlap_b4r3_mmx(cpln(bx, by), ppln(sx, sy), dpln(bx, by), thresh, inv_table, &weight);
            k++;
          }

          if (devn < thresh)
          {
            weight = get_weight(k);
            frcore_filter_overlap_b4r3_mmx(cpln(bx, by), npln(sx, sy), dpln(bx, by), thresh, inv_table, &weight);
          }
        }
      }
    }

    if ((P & 1))
      for (y = 2; y < dim.y - B; y += S)
      {
        int tmax = T;
        if (pl > 0) tmax = Tuv;

        R = 1;

        for (x = 2; x < dim.x - B; x += S)
        {
          int sx = x, sy = y;
          if (sx < R) sx = R;
          if (sy < R) sy = R;
          if (sx > dim.x - R - B) sx = dim.x - R - B;
          if (sy > dim.y - R - B) sy = dim.y - R - B;

          int dev = 10;
          frcore_dev_b4_mmx(cpln(sx, sy), &dev);
          int thresh = ((dev * lambda) >> 10);
          thresh = (thresh > tmax) ? tmax : thresh;
          if (thresh < 1) thresh = 1;

          int weight = get_weight(1);
          frcore_filter_diff_b4r1_mmx(cpln(x, y), cpln(sx, sy), dpln(x, y), thresh, inv_table, &weight);

          unsigned char* wpln_ptr = wpln.get_ptr(x / 4, y / 4);
          *wpln_ptr = clipb(weight);
        }
      }


    int k, kk;
    if ((P & 1))
      for (kk = 1; kk < 9; kk++)
      {
        k = kk;

        for (y = (k / 3) + 1; y < dim.y - B; y += S)
        {
          int tmax = T;
          if (pl > 0) tmax = Tuv;

          R = 2;

          for (x = (k % 3) + 1; x < dim.x - B; x += S)
          {
            int sx = x, sy = y;
            if (sx < R) sx = R;
            if (sy < R) sy = R;
            if (sx > dim.x - R - B) sx = dim.x - R - B;
            if (sy > dim.y - R - B) sy = dim.y - R - B;

            unsigned char* wpln_ptr = wpln.get_ptr(x / 4, y / 4);

            if (*wpln_ptr < P / 1000) continue;

            int dev = 10;
            frcore_dev_b4_mmx(cpln(sx, sy), &dev);

            int thresh = ((dev * lambda) >> 10);
            thresh = (thresh > tmax) ? tmax : thresh;
            if (thresh < 1) thresh = 1;

            int weight = get_weight(k);
            frcore_filter_overlap_b4r2_mmx(cpln(x, y), cpln(sx, sy), dpln(x, y), thresh, inv_table, &weight);
          }
        }

      }

  } // PLANES LOOP

  __asm emms

  return df;
}

AvsFilter::AvsFilter(AVSValue args, IScriptEnvironment* env)
  : GenericVideoFilter(args[0].AsClip())
{
  lastn = -2;

  lambda = (int)(args[1].AsFloat(1.1) * 1024);
  T = (int)(args[2].AsFloat(6) * 16);
  Tuv = (int)(args[3].AsFloat(2) * 16);

  P = args[4].AsInt(0);
  //	P = 12*1000;

  //	P &= ~7;
  //	P |= 1;		// adaptive overlapping
  //	P |= 2;		// temporal
  //	P |= 4;		// adaptive radius


  //	P = 12*1000+2;
  //	P |= 2;

  wpln = Plane8i().alloc(vi.width / 4, vi.height / 4);
  for (int i = 1; i < 1024; i++) inv_table[i] = (int)((1 << 15) / (double)i);
  inv_table[1] = 32767;
}

AvsFilter::~AvsFilter()
{
  wpln.free();
}

AVSValue __cdecl AvsFilter::Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
  return new AvsFilter(args, env);
}

// Declare and initialise server pointers static storage.
const AVS_Linkage* AVS_linkage = 0;

// DLL entry point called from LoadPlugin() to setup a user plugin.
extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors) {

  AVS_linkage = vectors;
  env->AddFunction("frfun7", "c[lambda]f[T]f[Tuv]f[P]i", AvsFilter::Create, 0);
  //    env->AddFunction("frfun7", "c[lambda]f[T]f[Tuv]f", AvsFilter::Create, 0);
  return "`x' xxx";
}
