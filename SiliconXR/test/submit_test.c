// GL readback check for Submit: two 8x4 eye textures go into a temp shm; the frame must be side-by-side, top row first, BGRA.
#define GL_SILENCE_DEPRECATION
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl3.h>
#include <assert.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include "../openvr_capi.h"
#include "../../common/vr4mac.h"

static GLuint tex(uint32_t top, uint32_t bottom) {   // RGBA words; GL row 0 is the image bottom
    uint32_t px[8 * 4];
    for (int i = 0; i < 32; i++) px[i] = i < 8 ? bottom : top;
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    return t;
}
int main(void) {
    char path[] = "/tmp/vr4mac_test_shm_XXXXXX";
    int fd = mkstemp(path); ftruncate(fd, VR4_SHM_SIZE);
    VR4Shm *s = mmap(NULL, VR4_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    s->magic = VR4_SHM_MAGIC; s->version = VR4_SHM_VERSION; s->eye_w = 8; s->eye_h = 4; s->fps = 500;
    setenv("VR4MAC_SHM", path, 1);

    CGLPixelFormatAttribute attrs[] = {kCGLPFAOpenGLProfile, (CGLPixelFormatAttribute)kCGLOGLPVersion_3_2_Core, 0};
    CGLPixelFormatObj pf; GLint n; CGLContextObj ctx;
    CGLChoosePixelFormat(attrs, &pf, &n); CGLCreateContext(pf, NULL, &ctx); CGLSetCurrentContext(ctx);

    void *lib = dlopen("../build/libopenvr_api.dylib", RTLD_NOW);
    assert(lib);
    intptr_t (*init)(EVRInitError *, EVRApplicationType) = dlsym(lib, "VR_InitInternal");
    void *(*get)(const char *, EVRInitError *) = dlsym(lib, "VR_GetGenericInterface");
    EVRInitError err; init(&err, EVRApplicationType_VRApplication_Scene); assert(err == 0);
    struct VR_IVRCompositor_FnTable *c = get("FnTable:IVRCompositor_027", &err);
    TrackedDevicePose_t poses[3];
    c->WaitGetPoses(poses, 3, NULL, 0);
    const uint32_t RED = 0xff0000ffu, BLUE = 0xffff0000u, GREEN = 0xff00ff00u, WHITE = 0xffffffffu;   // RGBA bytes, little endian
    Texture_t l = {(void *)(uintptr_t)tex(RED, BLUE), ETextureType_TextureType_OpenGL, EColorSpace_ColorSpace_Auto};
    Texture_t r = {(void *)(uintptr_t)tex(GREEN, WHITE), ETextureType_TextureType_OpenGL, EColorSpace_ColorSpace_Auto};
    assert(c->Submit(EVREye_Eye_Left, &l, NULL, 0) == 0 && s->frame_seq == 0);
    assert(c->Submit(EVREye_Eye_Right, &r, NULL, 0) == 0 && s->frame_seq == 1);
    assert(s->frame_w[1] == 16 && s->frame_h[1] == 4 && s->frame_rgba[1] == 0);
    uint8_t *f = vr4_frame(s, 1);
    #define PX(x, y) (f + ((y) * 16 + (x)) * 4)
    assert(PX(0, 0)[2] == 255 && PX(0, 0)[0] == 0);     // left eye top row red (B,G,R,A)
    assert(PX(7, 3)[0] == 255 && PX(7, 3)[2] == 0);     // left eye bottom row blue
    assert(PX(8, 0)[1] == 255 && PX(8, 0)[0] == 0);     // right eye top row green
    assert(PX(15, 3)[0] == 255 && PX(15, 3)[2] == 255); // right eye bottom row white
    assert(glGetError() == GL_NO_ERROR);
    unlink(path);
    puts("PASS submit");
    return 0;
}
