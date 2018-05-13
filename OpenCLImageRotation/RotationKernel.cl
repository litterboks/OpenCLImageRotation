// define a pixel struct to make the code cleaner
typedef struct tag_Pixel
{
	unsigned char R;
	unsigned char G;
	unsigned char B;
} Pixel;

__kernel void image_rotate(	__global Pixel* src_data, __global Pixel* dest_data, int W, int H, float sinTheta, float cosTheta)
{
	// get the global ids
	const int ix = get_global_id(0);
	const int iy = get_global_id(1);

	// use the equation to get the new coordinates
	float xpos = (((float)ix - W / 2)*cosTheta - ((float)iy - H / 2)*sinTheta) + (W / 2);
	float ypos = (((float)ix - W / 2)*sinTheta + ((float)iy - H / 2)*cosTheta) + (H / 2);

	// assign the new coordinates
	if ((((int)xpos >= 0) && ((int)xpos < W)) && (((int)ypos >= 0) && ((int)ypos < H)))
	{
		dest_data[iy*W + ix] = src_data[(int)floor(ypos*W + xpos)];
	}
}
