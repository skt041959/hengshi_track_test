#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/mman.h>

int main (int argc, char* argv[])
{
  int fp=0;
  struct fb_var_screeninfo vinfo;
  struct fb_fix_screeninfo finfo;
  long screensize=0;
  char *fbp = 0;
  int x = 0, y = 0;
  long location = 0;
  int i;

  unsigned char col[] = {255u, 255u, 255u, 255u};

  fp = open ("/dev/fb0",O_RDWR);

  if (fp < 0){
    printf("Error : Can not open framebuffer device\n");
    exit(1);
  }

  if (ioctl(fp,FBIOGET_FSCREENINFO,&finfo)){
    printf("Error reading fixed information\n");
    exit(2);
  }

  if (ioctl(fp,FBIOGET_VSCREENINFO,&vinfo)){
    printf("Error reading variable information\n");
    exit(3);
  }

  screensize = vinfo.xres * vinfo.yres * vinfo.bits_per_pixel / 8;
  /*这就是把fp所指的文件中从开始到screensize大小的内容给映射出来，得到一个指向这块空间的指针*/
  fbp =(char *) mmap (0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fp, 0);

  if ((int) fbp == -1)
  {
    printf ("Error: failed to map framebuffer device to memory.\n");
    exit (4);
  }

  /*这是你想画的点的位置坐标,(0，0)点在屏幕左上角*/
  x = atoi(argv[1]);
  y = atoi(argv[2]);

  location = x * (vinfo.bits_per_pixel / 8) + y  *  finfo.line_length;
  //location = 0;
  //

  unsigned int j = 0x3ffffu;

  while(j--)
  {
    for(i=0; i<200; i++)
    {
      memcpy(fbp+location+i*4, col, 4);
    }
  }

  munmap (fbp, screensize);	/*解除映射*/

  close (fp);				/*关闭文件*/
  return 0;

}
