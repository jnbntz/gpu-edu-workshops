/*
 * Copyright (c) 2007 Yiorgos Adamopoulos <adamo@dblab.ece.ntua.gr>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

/* 
 * Altered by Jonathan Bentz (NVIDIA), 2015.  
 * Changes involve the ability to print the labels separate from the 
 * images for input into other C programs.
 */

/* This program is written on an OpenBSD machine. This means that you most
 * likely need to change the betoh32() calls to ntohl(). Uncomment the next
 * macro if that is the case.
 */
#define betoh32 ntohl /* */

#include <stdio.h>
#include <unistd.h>
#include <sysexits.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <math.h>

extern int optind;
extern char *optarg;
extern int errno;

#define PNAME_MAX       72

static char _progname[] = "mnist";

void
usage(char *s)
{
        fprintf(stderr, "usage: %s [-h] | [-0123456789] [-l label.idx] [-i image.idx] [-b basename] [-p] [-r]\n\n", s);
        fprintf(stderr, "\t-h : this help message.\n");
        fprintf(stderr, "\t-l : idx file listing the labels.\n");
        fprintf(stderr, "\t-i : idx file containing the image descriptions.\n");
        fprintf(stderr, "\t-p : extact to PGM files.\n");
        fprintf(stderr, "\t-b : basename for the PGM files.\n");
        fprintf(stderr, "\t-r : reverse video (used together with -p).\n");
        fprintf(stderr, "\t-0 .. -9 : Label / Pattern to extract from idx image file.\n");
        fprintf(stderr, "\nIf -p is not specified, the array values are printed on stdout separated by a blank line.\n");

        exit(EX_USAGE);
}

u_char
get_next_label(int f, u_int32_t n)
{
        int c;
        u_char byte;

        c = read(f, &byte, sizeof(byte));
        if (c < 0) {
                fprintf(stderr, "read label %d: %s\n", n, strerror(errno));
                exit(EX_IOERR);
        }

        return(byte);
}

void
print_pgm_image(int f, u_int32_t n, u_char *image, u_int32_t nr, u_int32_t nc, char *basename, int r_flag)
{
        int c, ilen, i, j;
        FILE *fp;
        char pname[PNAME_MAX];
        u_char w;

        ilen = nr * nc;
        c = read(f, image, ilen);
        if (c < 0) {
                fprintf(stderr, "read image %u: %s\n", n, strerror(errno));
                exit(EX_IOERR);
        }

        snprintf(pname, PNAME_MAX, "%s-%d.pgm", basename, n);
        fp = fopen(pname, "w");
        if (fp == NULL) {
                fprintf(stderr, "fopen(%s): %s", pname, strerror(errno));
                exit(EX_IOERR);
        }

        fprintf(fp, "P5\n");
        fprintf(fp, "# %s\n", pname);
        fprintf(fp, "%u %u\n", nr, nc);
        fprintf(fp, "255\n");
        fflush(fp);

        for (i = 0; i < nr; i++) {
                for (j = 0; j < nc; j++) {
                        w = *(image + j + (i * nc));
                        if (r_flag) {
                                w = 255 - w;
                        }
                        write(fileno(fp), &w, 1);
                }
                fprintf(fp, "\n");
        }
        fprintf(fp, "\n");

        fclose(fp);
        return;
}

void
print_next_image(int f, u_int32_t n, u_char *image, u_int32_t nr, u_int32_t nc)
{
        int c, ilen, i, j;

        ilen = nr * nc;
        c = read(f, image, ilen);
        if (c < 0) {
                fprintf(stderr, "read image %u: %s\n", n, strerror(errno));
                exit(EX_IOERR);
        }

        /* similar to the PGM format; we print the width and height of
         * the array first
         */
//      printf("%u %u\n", nr, nc);
        for (i = 0; i < nr; i++) {
                for (j = 0; j < nc; j++) {
                        printf(" %u", *(image + j + (i * nc)));
                }
//              printf("\n");
        }
        printf("\n");

        return;
}

int
main(int argc, char **argv)
{
        int c, n_flag, l_flag, i_flag, b_flag, p_flag, r_flag;
        int fl, fi;
        u_int32_t i, j, lmagic, imagic, lnum, inum, nr, nc, ilen, n;
        u_char byte, *image;
        char *lfile, *ifile, *basename;

        n_flag = -1;
        l_flag = i_flag = b_flag = p_flag = r_flag = 0;
        while((c = getopt(argc, argv, "h0123456789l:i:b:pr")) != -1) {
                switch(c) {
                case '0':
                        n_flag = 0;
                        break;
                case '1':
                        n_flag = 1;
                        break;
                case '2':
                        n_flag = 2;
                        break;
                case '3':
                        n_flag = 3;
                        break;
                case '4':
                        n_flag = 4;
                        break;
                case '5':
                        n_flag = 5;
                        break;
                case '6':
                        n_flag = 6;
                        break;
                case '7':
                        n_flag = 7;
                        break;
                case '8':
                        n_flag = 8;
                        break;
                case '9':
                        n_flag = 9;
                        break;
                case 'l':
                        lfile = optarg;
                        l_flag++;
                        break;
                case 'i':
                        ifile = optarg;
                        i_flag++;
                        break;
                case 'b':
                        basename = optarg;
                        b_flag++;
                        break;
                case 'p':
                        p_flag++;
                        break;
                case 'r':
                        r_flag++;
                        break;
                case 'h':
                default:
                        usage(_progname);
                        /* NOTREACHED */
                }
        }
        argc -= optind;
        argv += optind;

        /* We must at least pick a number between 0..9 */
        if (n_flag < 0) {
                fprintf(stderr, "Please select a number to be extracted!\n");
                usage(_progname);
        }

        if (!l_flag) {
                fprintf(stderr, "Please specify a label idx file!\n");
                usage(_progname);
        }

        if (!i_flag) {
                fprintf(stderr, "Please specify an image idx file!\n");
                usage(_progname);
        }

        if (p_flag) {
                if (!b_flag) {
                        fprintf(stderr, "You must specify a basename!\n");
                        usage(_progname);
                }
        }

        fl = open(lfile, O_RDONLY, 0);
        if (fl < 0) {
                fprintf(stderr, "open(%s): %s\n", lfile, strerror(errno));
                exit(EX_USAGE);
        }

        fi = open(ifile, O_RDONLY, 0);
        if (fi < 0) {
                fprintf(stderr, "open(%s): %s\n", ifile, strerror(errno));
                exit(EX_USAGE);
        }

        /* get the magic numbers */
        c = read(fl, &lmagic, sizeof(lmagic));
        if (c < 0) {
                fprintf(stderr, "read(lmagic): %s\n", strerror(errno));
                exit(EX_IOERR);
        }
        lmagic = betoh32(lmagic); /* idx files are big endian */

        c = read(fi, &imagic, sizeof(imagic));
        if (c < 0) {
                fprintf(stderr, "read(imagic): %s\n", strerror(errno));
                exit(EX_IOERR);
        }
        imagic = betoh32(imagic); /* idx files are big endian */

        /* read the number of items */
        c = read(fl, &lnum, sizeof(lnum));
        if (c < 0) {
                fprintf(stderr, "read(lnum): %s\n", strerror(lnum));
                exit(EX_IOERR);
        }
        lnum = betoh32(lnum);

        c = read(fi, &inum, sizeof(inum));
        if (c < 0) {
                fprintf(stderr, "read(inum): %s\n", strerror(lnum));
                exit(EX_IOERR);
        }
        inum = betoh32(inum);

        /* At the very least lnum and inum are equal */
        if (lnum != inum) {
                fprintf(stderr, "Please use label and image files that at least have an equal number of items!\n");
                usage(_progname);
        }

        /* read the number of rows */
        c = read(fi, &nr, sizeof(nr));
        if (c < 0) {
                fprintf(stderr, "read number of rows: %s\n", strerror(errno));
                exit(EX_IOERR);
        }
        nr = betoh32(nr);

        /* read the number of columns */
        c = read(fi, &nc, sizeof(nc));
        if (c < 0) {
                fprintf(stderr, "read number of columns: %s\n", strerror(errno));
                exit(EX_IOERR);
        }
        nc = betoh32(nc);

        /* allocate the image buffer */
        ilen = nr * nc;
        image = (u_char *)malloc(ilen);
        if (image == NULL) {
                fprintf(stderr, "malloc(image): %s\n", strerror(errno));
                exit(EX_OSERR);
        }

        #if 0
        printf("magic numbers: %u %u\n", lmagic, imagic);
        printf("#items: %u %u\n", lnum, inum);
        printf("#rows: %u #columns: %u\n", nr, nc);
        #endif

        n = 0;
        while (n != lnum) {
                byte = get_next_label(fl, n);

//jlb              if (byte == n_flag) {
                if (byte == byte) {
//jlb                printf("%d\n",byte);
// printing the digit number
                   fprintf(stderr,"%d\n",byte);
                        if (p_flag) {
                                print_pgm_image(fi, n, image, nr, nc, basename, r_flag);
                        } else {
                                print_next_image(fi, n, image, nr, nc);
                        }
                } else {
                        if (lseek(fi, ilen, SEEK_CUR) < 0) {
                                fprintf(stderr, "lseek image %u: %s", n, strerror(errno));
                                exit(EX_IOERR);
                        }
                }

                n++;
        }
        close(fi);
        close(fl);

        exit(EX_OK);
}

