#include <iostream>
#include <cstdlib>
#include <cmath>

#define RGB_MAX_VALUE 255
#define HUE_MAX_VALUE 360

// adapted from http://www.easyrgb.com/en/math.php
void RGBtoXYZ(double R, double G, double B, double& X, double& Y, double& Z)
{
	// sR, sG and sB (Standard RGB) input range = 0 ÷ 255
	// X, Y and Z output refer to a D65/2° standard illuminant.

	R /= RGB_MAX_VALUE;
	G /= RGB_MAX_VALUE;
	B /= RGB_MAX_VALUE;

	if (R > 0.04045)	{ R = pow((R + 0.055) / 1.055, 2.4); }
	else				{ R /= 12.92; }

	if (G > 0.04045)	{ G = pow((G + 0.055) / 1.055, 2.4); }
	else				{ G /= 12.92; }

	if (B > 0.04045)	{ B = pow((B + 0.055) / 1.055, 2.4); }
	else				{ B /= 12.92; }

	//R *= 100;
	//G *= 100;
	//B *= 100;

	X = R * 0.4124 + G * 0.3576 + B * 0.1805;
	Y = R * 0.2126 + G * 0.7152 + B * 0.0722;
	Z = R * 0.0193 + G * 0.1192 + B * 0.9505;
}

// adapted from http://www.easyrgb.com/en/math.php
void XYZtoRGB(double X, double Y, double Z, double& R, double& G, double& B)
{
	//X, Y and Z input refer to a D65/2° standard illuminant.
	//sR, sG and sB (standard RGB) output range = 0 ÷ 255

	//X /= 100;
	//Y /= 100;
	//Z /= 100;

	R = X *  3.2406 + Y * -1.5372 + Z * -0.4986;
	G = X * -0.9689 + Y *  1.8758 + Z *  0.0415;
	B = X *  0.0557 + Y * -0.2040 + Z *  1.0570;

	if (R > 0.0031308)	{ R = 1.055 * pow(R, ( 1 / 2.4 )) - 0.055; }
	else				{ R *= 12.92; }

	if (G > 0.0031308)	{ G = 1.055 * pow(G, ( 1 / 2.4 )) - 0.055; }
	else				{ G *= 12.92; }

	if (B > 0.0031308)	{ B = 1.055 * pow(B, ( 1 / 2.4 )) - 0.055; }
	else				{ B *= 12.92; }

	R *= RGB_MAX_VALUE;
	G *= RGB_MAX_VALUE;
	B *= RGB_MAX_VALUE;
}

/*void XYZtoCIELUV(double X, double Y, double Z, double& L, double& U, double& V) {
	//Reference-X, Y and Z refer to specific illuminants and observers.
	//Common reference values are available below in this same page.

	var_U = (4 * X) / (X + (15 * Y) + (3 * Z));
	var_V = (9 * Y) / (X + (15 * Y) + (3 * Z));

	var_Y = Y / 100;
	if (var_Y > 0.008856) var_Y = var_Y ^ (1 / 3);
	else var_Y = (7.787 * var_Y) + (16 / 116);

	ref_U = (4 * Reference - X) / (Reference - X + (15 * Reference - Y) + (3 * Reference - Z));
	ref_V = (9 * Reference - Y) / (Reference - X + (15 * Reference - Y) + (3 * Reference - Z));

	CIE - L * = (116 * var_Y) - 16;
	CIE - u * = 13 * CIE - L * *(var_U - ref_U);
	CIE - v * = 13 * CIE - L * *(var_V - ref_V);
}*/

/*! \brief Convert RGB to HSV color space

  Converts a given set of RGB values `r', `g', `b' into HSV
  coordinates. The input RGB values are in the range [0, 1], and the
  output HSV values are in the ranges h = [0, 360], and s, v = [0,
  1], respectively.

  \param fR Red component, used as input, range: [0, 1]
  \param fG Green component, used as input, range: [0, 1]
  \param fB Blue component, used as input, range: [0, 1]
  \param fH Hue component, used as output, range: [0, 360]
  \param fS Hue component, used as output, range: [0, 1]
  \param fV Hue component, used as output, range: [0, 1]

*/
void RGBtoHSV(float fR, float fG, float fB, float& fH, float& fS, float& fV) {
    float fCMax = max(max(fR, fG), fB);
    float fCMin = min(min(fR, fG), fB);
    float fDelta = fCMax - fCMin;

    if(fDelta > 0) {
        if(fCMax == fR) {
            fH = 60 * (fmod(((fG - fB) / fDelta), 6));
        } else if(fCMax == fG) {
            fH = 60 * (((fB - fR) / fDelta) + 2);
        } else if(fCMax == fB) {
            fH = 60 * (((fR - fG) / fDelta) + 4);
        }

        if(fCMax > 0) {
            fS = fDelta / fCMax;
        } else {
            fS = 0;
        }

        fV = fCMax;
    } else {
        fH = 0;
        fS = 0;
        fV = fCMax;
    }

    if(fH < 0) {
        fH = 360 + fH;
    }
}


/*! \brief Convert HSV to RGB color space

  Converts a given set of HSV values `h', `s', `v' into RGB
  coordinates. The output RGB values are in the range [0, 1], and
  the input HSV values are in the ranges h = [0, 360], and s, v =
  [0, 1], respectively.

  \param fR Red component, used as output, range: [0, 1]
  \param fG Green component, used as output, range: [0, 1]
  \param fB Blue component, used as output, range: [0, 1]
  \param fH Hue component, used as input, range: [0, 360]
  \param fS Hue component, used as input, range: [0, 1]
  \param fV Hue component, used as input, range: [0, 1]

*/
void HSVtoRGB(float& fR, float& fG, float& fB, float fH, float fS, float fV) {
    float fC = fV * fS; // Chroma
    float fHPrime = fmod(fH / 60.0, 6);
    float fX = fC * (1 - fabs(fmod(fHPrime, 2) - 1));
    float fM = fV - fC;

    if(0 <= fHPrime && fHPrime < 1) {
        fR = fC;
        fG = fX;
        fB = 0;
    } else if(1 <= fHPrime && fHPrime < 2) {
        fR = fX;
        fG = fC;
        fB = 0;
    } else if(2 <= fHPrime && fHPrime < 3) {
        fR = 0;
        fG = fC;
        fB = fX;
    } else if(3 <= fHPrime && fHPrime < 4) {
        fR = 0;
        fG = fX;
        fB = fC;
    } else if(4 <= fHPrime && fHPrime < 5) {
        fR = fX;
        fG = 0;
        fB = fC;
    } else if(5 <= fHPrime && fHPrime < 6) {
        fR = fC;
        fG = 0;
        fB = fX;
    } else {
        fR = 0;
        fG = 0;
        fB = 0;
    }

    fR += fM;
    fG += fM;
    fB += fM;
}

void _RGBtoHSV(int R, int G, int B, float& fH, float& fS, float& fV) {
	float fR = (float) R / RGB_MAX_VALUE;
	float fG = (float) G / RGB_MAX_VALUE;
	float fB = (float) B / RGB_MAX_VALUE;

	RGBtoHSV(fR, fG, fB, fH, fS, fV);
	fH /= HUE_MAX_VALUE; // hue must be in [0, 1]
}

void _HSVtoRGB(int& R, int& G, int& B, float fH, float fS, float fV) {
	float fR, fG, fB;
	HSVtoRGB(fR, fG, fB, fH, fS, fV);

	R = (int) (fR * RGB_MAX_VALUE);
	G = (int) (fG * RGB_MAX_VALUE);
	B = (int) (fB * RGB_MAX_VALUE);
}