Optional image component
==========================

Headers: ``src/images/KokkosDNN_Image*.hpp``. Built only with
``KokkosDNN_ENABLE_IMAGE=ON`` (stb-backed JPEG/PNG/BMP decode plus
PNG/JPEG encode and resize/normalize preprocessing into device views;
WebP decode plus WebP encode via libwebp unless
``KokkosDNN_ENABLE_IMAGE_WEBP=OFF``). See
``src/images/KokkosDNN_ImageIO.hpp``,
``src/images/KokkosDNN_ImagePreprocess.hpp``, and
``src/images/KokkosDNN_ImageTypes.hpp``.
