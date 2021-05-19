#include "avisynth.h"
#include "Frfun7.h"

#include <math.h>
#include <algorithm>

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

void frcore_filter_b4r3_mmx(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int T, int* inv_table, int* weight)
{
  ptra += -3 * pitcha - 3; // cpln(-3, -3)

  int* _weight = weight;
  int* _inv_table = inv_table;
  int _thresh = T;

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

void frcore_filter_adapt_b4r3_mmx(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int T, int sT2, int sT3, int* inv_table, int* weight)
{

  ptra += -1 * pitcha - 3; // cpln(-3, -1)

  int* _weight = weight;
  int* _inv_table = inv_table;
  int _thresh = T;
  int _thresh2 = sT2;
  int _thresh3 = sT3;

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

void frcore_filter_b4r0_mmx(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int T, int* inv_table, int* weight)
{

  // ptra no change // cpln(-0, -0)

  int* _weight = weight;
  int* _inv_table = inv_table;
  int _thresh = T;

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

#define expand_word(mmA,mmB)				\
__asm			movq	mmB, mmA			\
__asm			psllq	mmA, 32				\
__asm			por		mmA, mmB			\
__asm			movq	mmB, mmA			\
__asm			psllq	mmB, 16				\
__asm			por		mmA, mmB


void frcore_filter_overlap_b4r3_mmx(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int T, int* inv_table, int* weight)
{

  ptra += -3 * pitcha - 3; // cpln(-3, -3)

  int* _weight = weight;
  int* _inv_table = inv_table;
  int _thresh = T;

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

void frcore_filter_overlap_b4r2_mmx(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int T, int* inv_table, int* weight)
{

  ptra += -2 * pitcha - 2; // cpln(-2, -2)

  int* _weight = weight;
  int* _inv_table = inv_table;
  int _thresh = T;

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

void frcore_filter_overlap_b4r0_mmx(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int T, int* inv_table, int* weight)
{

  // ptra no change // cpln(0, 0)

  int* _weight = weight;
  int* _inv_table = inv_table;
  int _thresh = T;

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


void frcore_filter_diff_b4r1_mmx(const uint8_t* ptrr, int pitchr, const uint8_t* ptra, int pitcha, uint8_t* ptrb, int pitchb, int T, int* inv_table, int* weight)
{

  ptra += -1 * pitcha - 1; //  cpln(-1, -1)

  int* _weight = weight;
  int* _inv_table = inv_table;
  int _thresh = T;

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

#if 0
// not used
void frcore_filter_b4r2_mmx(Plane8i rpln, Plane8i cpln, uint8_t* dstp, int dstp_pitch, int T, int* inv_table)
{

  uint8_t* ptrr = rpln.ptr;
  uint8_t* ptra = cpln(-2, -2).ptr;
  uint8_t* ptrb = dstp;

  int* _inv_table = inv_table;
  int _thresh = T;
  int pitchr = rpln.stride;
  int pitcha = cpln.stride;
  int pitchb = dstp_pitch;

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
#endif

void frcore_dev_b4_mmx(const uint8_t* srcp, int src_pitch, int* dev)
{

  const uint8_t* ptra = srcp - 1; //  cpln(-1, 0).ptr;
  int pitcha = src_pitch;
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

void frcore_sad_b4_mmx(const uint8_t* ptra, int pitcha, const uint8_t* ptrb, int pitchb, int* sad)
{
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

int clipb(int weight) {
  return weight < 0 ? 0 : weight > 255 ? 255 : weight;
}

PVideoFrame __stdcall AvsFilter::GetFrame(int n, IScriptEnvironment* env)
{
  const bool mode_adaptive_overlapping = P & 1;
  const bool mode_temporal = P & 2;
  const bool mode_adaptive_radius = P & 4;

  // fixme: why was it uncommented?
  //	if (n<1 || n>vi.num_frames-1) return child->GetFrame(n, env);

  PVideoFrame pf; // previous
  PVideoFrame nf; // next

  PVideoFrame cf = child->GetFrame(n, env);

  if (mode_temporal) {
    pf = child->GetFrame(n - 1, env);
    nf = child->GetFrame(n + 1, env);
  }
  PVideoFrame df = env->NewVideoFrame(vi); // destination

  const int num_of_planes = std::min(vi.NumComponents(), 3);
  for (int pl = 0; pl < num_of_planes; pl++) { // PLANES LOOP
    const int plane = pl == 0 ? PLANAR_Y : (pl == 1 ? PLANAR_U : PLANAR_V);

    const int dim_x = cf->GetRowSize(plane);
    const int dim_y = cf->GetHeight(plane);;

    // prev/next: only for temporal
    const uint8_t* srcp_prev_orig = nullptr;
    const uint8_t* srcp_next_orig = nullptr;
    int src_prev_pitch;
    int src_next_pitch;

    if (mode_temporal) {
      srcp_prev_orig = pf->GetReadPtr(plane);
      src_prev_pitch = pf->GetPitch(plane);

      srcp_next_orig = nf->GetReadPtr(plane);
      src_next_pitch = nf->GetPitch(plane);
    }

    const uint8_t* srcp_orig = cf->GetReadPtr(plane);
    const int src_pitch = cf->GetPitch(plane);

    uint8_t* dstp_orig = df->GetWritePtr(plane);
    const int dstp_pitch = df->GetPitch(plane);

    int tmax = Thresh_luma;
    if (pl > 0) tmax = Thresh_chroma;

    int R = 3;
    int B = 4;
    int S = 4;
    int W = R * 2 + 1;

    for (int y = 0; y < dim_y + B - 1; y += S)
    {
      for (int x = 0; x < dim_x + B - 1; x += S)
      {
        int sx = x, sy = y;
        int bx = x, by = y;
        if (sx < R) sx = R;
        if (sy < R) sy = R;
        if (sx > dim_x - R - B) sx = dim_x - R - B;
        if (sy > dim_y - R - B) sy = dim_y - R - B;
        if (bx > dim_x - B) bx = dim_x - B;
        if (by > dim_y - B) by = dim_y - B;

        uint8_t* dstp = dstp_orig + dstp_pitch * by + bx;
        const uint8_t* srcp_s = srcp_orig + src_pitch * sy + sx; // cpln(sx, sy)

        int dev, devp, devn;
        frcore_dev_b4_mmx(srcp_s, src_pitch, &dev);

        // only for temporal use
        const uint8_t* srcpn_s = nullptr;
        const uint8_t* srcpp_s = nullptr;

        if (mode_temporal)
        {
          srcpp_s = srcp_prev_orig + src_prev_pitch * sy + sx; // ppln(sx, sy)
          frcore_sad_b4_mmx(srcp_s, src_pitch, srcpp_s, src_prev_pitch, &devp);

          srcpn_s = srcp_next_orig + src_next_pitch * sy + sx; // npln(sx, sy)
          frcore_sad_b4_mmx(srcp_s, src_pitch, srcpn_s, src_next_pitch, &devn);

          dev = std::min(dev, devn);
          dev = std::min(dev, devp);
        }

        int thresh = ((dev * lambda) >> 10);
        thresh = (thresh > tmax) ? tmax : thresh;
        if (thresh < 1) thresh = 1;

        const uint8_t* srcp_b = srcp_orig + src_pitch * by + bx; // cpln(bx, by)

        int weight;
        if (mode_temporal) {
          frcore_filter_b4r0_mmx(srcp_b, src_pitch, srcp_b, src_pitch, dstp, dstp_pitch, thresh, inv_table, &weight);

          int k = 1;
          if (devp < thresh)
          {
            weight = get_weight(k);
            frcore_filter_overlap_b4r3_mmx(srcp_b, src_pitch, srcpp_s, src_prev_pitch, dstp, dstp_pitch, thresh, inv_table, &weight);
            k++;
          }

          if (devn < thresh)
          {
            weight = get_weight(k);
            frcore_filter_overlap_b4r3_mmx(srcp_b, src_pitch, srcpn_s, src_next_pitch, dstp, dstp_pitch, thresh, inv_table, &weight);
          }
        }
        else
        {
          // not temporal
          if (sx == x && sy == y && mode_adaptive_radius)
            frcore_filter_adapt_b4r3_mmx(srcp_b, src_pitch, srcp_s, src_pitch, dstp, dstp_pitch, thresh, 16 * 9, 16 * 25, inv_table, &weight);
          else // Nothing or adaptive_overlapping or some case of adaptive_radius
            frcore_filter_b4r3_mmx(srcp_b, src_pitch, srcp_s, src_pitch, dstp, dstp_pitch, thresh, inv_table, &weight);
        }

      }
    }

    if (mode_adaptive_overlapping)
    {
      for (int y = 2; y < dim_y - B; y += S)
      {

        R = 1;

        for (int x = 2; x < dim_x - B; x += S)
        {
          int sx = x, sy = y;
          if (sx < R) sx = R;
          if (sy < R) sy = R;
          if (sx > dim_x - R - B) sx = dim_x - R - B;
          if (sy > dim_y - R - B) sy = dim_y - R - B;

          int dev = 10;
          const uint8_t* srcp_s = srcp_orig + src_pitch * sy + sx; // cpln(sx, sy)
          frcore_dev_b4_mmx(srcp_s, src_pitch, &dev);

          int thresh = ((dev * lambda) >> 10);
          thresh = (thresh > tmax) ? tmax : thresh;
          if (thresh < 1) thresh = 1;

          const uint8_t* srcp_xy = srcp_orig + src_pitch * y + x; // cpln(x, y)
          uint8_t* dstp = dstp_orig + dstp_pitch * y + x;

          int weight = get_weight(1);
          frcore_filter_diff_b4r1_mmx(srcp_xy, src_pitch, srcp_s, src_pitch, dstp, dstp_pitch, thresh, inv_table, &weight);

          unsigned char* wpln_ptr = wpln + wp_stride * (y / 4) + (x / 4);
          *wpln_ptr = clipb(weight);
        }
      }

      for (int kk = 1; kk < 9; kk++)
      {
        int k = kk;

        for (int y = (k / 3) + 1; y < dim_y - B; y += S)
        {
          R = 2;

          for (int x = (k % 3) + 1; x < dim_x - B; x += S)
          {
            int sx = x, sy = y;
            if (sx < R) sx = R;
            if (sy < R) sy = R;
            if (sx > dim_x - R - B) sx = dim_x - R - B;
            if (sy > dim_y - R - B) sy = dim_y - R - B;

            unsigned char* wpln_ptr = wpln + wp_stride * (y / 4) + (x / 4);
            if (*wpln_ptr < P1_param) continue;

            int dev = 10;
            const uint8_t* srcp = srcp_orig + src_pitch * sy + sx; // cpln(sx, sy)
            frcore_dev_b4_mmx(srcp, src_pitch, &dev);

            int thresh = ((dev * lambda) >> 10);
            thresh = (thresh > tmax) ? tmax : thresh;
            if (thresh < 1) thresh = 1;

            uint8_t* dstp = dstp_orig + dstp_pitch * y + x;
            const uint8_t* srcp_s = srcp_orig + src_pitch * sy + sx; // cpln(sx, sy)
            const uint8_t* srcp_xy = srcp_orig + src_pitch * y + x; // cpln(x, y)
            int weight = get_weight(k);
            frcore_filter_overlap_b4r2_mmx(srcp_xy, src_pitch, srcp_s, src_pitch, dstp, dstp_pitch, thresh, inv_table, &weight);
          }
        }

      }
    } // adaptive overlapping
  } // PLANES LOOP

  __asm emms

  return df;
}

AvsFilter::AvsFilter(AVSValue args, IScriptEnvironment* env)
  : GenericVideoFilter(args[0].AsClip())
{
  lambda = (int)(args[1].AsFloat(1.1f) * 1024); // 10 bit integer arithmetic
  // parameter "T"
  Thresh_luma = (int)(args[2].AsFloat(6) * 16);
  // parameter "Tuv"
  Thresh_chroma = (int)(args[3].AsFloat(2) * 16);
  // parameter "P"
  const int P_param = args[4].AsInt(0);

  P = P_param & 7;
  P1_param = P == 1 ? P_param / 1000 : 0; // hidden parameter used only for adaptive overlapping

  // for adaptive overlapping: a number*1000 makes a weight threshold (?)
  //	P = 12*1000;
  //	P &= ~7;
  //	P |= 1;		// adaptive overlapping
  //	P |= 2;		// temporal
  //	P |= 4;		// adaptive radius

  wp_width = vi.width / 4;
  wp_height = vi.height / 4;
  const int ALIGN = 32;
  wp_stride = (((wp_width)+(ALIGN)-1) & (~((ALIGN)-1)));
  if (P & 1) // only used for adaptive
    wpln = new uint8_t[wp_stride * wp_height];
  else
    wpln = nullptr;

  // pre-build reciprocial table
  for (int i = 1; i < 1024; i++) {
    // 1/x table 1..1023 for 15 bit integer arithmetic
    inv_table[i] = (int)((1 << 15) / (double)i);
  }
  inv_table[1] = 32767; // 2^15 - 1
}

AvsFilter::~AvsFilter()
{
  delete[] wpln;
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
  // this P parameter did not exist in rev6 but appeared in a build in 2013
  env->AddFunction("frfun7", "c[lambda]f[T]f[Tuv]f[P]i", AvsFilter::Create, 0);
  //    env->AddFunction("frfun7", "c[lambda]f[T]f[Tuv]f", AvsFilter::Create, 0);
  return "`x' xxx";
}
