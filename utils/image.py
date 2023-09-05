import cv2
import base64
import numpy as np


def laplacian_blending(A, B, m, num_levels=7):
    assert A.shape == B.shape
    assert B.shape == m.shape
    height = m.shape[0]
    width = m.shape[1]
    size_list = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    size = size_list[np.where(size_list > max(height, width))][0]
    GA = np.zeros((size, size, 3), dtype=np.float32)
    GA[:height, :width, :] = A
    GB = np.zeros((size, size, 3), dtype=np.float32)
    GB[:height, :width, :] = B
    GM = np.zeros((size, size, 3), dtype=np.float32)
    GM[:height, :width, :] = m
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))
    lpA  = [gpA[num_levels-1]]
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in range(num_levels-1,0,-1):
        LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1])
    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)
    ls_ = LS[0]
    for i in range(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    ls_ = ls_[:height, :width, :]
    #ls_ = (ls_ - np.min(ls_)) * (255.0 / (np.max(ls_) - np.min(ls_)))
    return ls_.clip(0, 255)


def mask_crop(mask, crop):
    top, bottom, left, right = crop
    shape = mask.shape
    top = int(top)
    bottom = int(bottom)
    if top + bottom < shape[1]:
        if top > 0: mask[:top, :] = 0
        if bottom > 0: mask[-bottom:, :] = 0

    left = int(left)
    right = int(right)
    if left + right < shape[0]:
        if left > 0: mask[:, :left] = 0
        if right > 0: mask[:, -right:] = 0

    return mask

def create_image_grid(images, size=128):
    num_images = len(images)
    num_cols = int(np.ceil(np.sqrt(num_images)))
    num_rows = int(np.ceil(num_images / num_cols))
    grid = np.zeros((num_rows * size, num_cols * size, 3), dtype=np.uint8)

    for i, image in enumerate(images):
        row_idx = (i // num_cols) * size
        col_idx = (i % num_cols) * size
        image = cv2.resize(image.copy(), (size,size))
        if image.dtype != np.uint8:
            image = (image.astype('float32') * 255).astype('uint8')
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        grid[row_idx:row_idx + size, col_idx:col_idx + size] = image

    return grid


def paste_to_whole(foreground, background, matrix, mask=None, crop_mask=(0,0,0,0), blur_amount=0.1, erode_amount = 0.15, blend_method='linear'):
    inv_matrix = cv2.invertAffineTransform(matrix)
    fg_shape = foreground.shape[:2]
    bg_shape = (background.shape[1], background.shape[0])
    foreground = cv2.warpAffine(foreground, inv_matrix, bg_shape, borderValue=0.0, borderMode=cv2.BORDER_REPLICATE)

    if mask is None:
        mask = np.full(fg_shape, 1., dtype=np.float32)
        mask = mask_crop(mask, crop_mask)
        mask = cv2.warpAffine(mask, inv_matrix, bg_shape, borderValue=0.0)
    else:
        assert fg_shape == mask.shape[:2], "foreground & mask shape mismatch!"
        mask = mask_crop(mask, crop_mask).astype('float32')
        mask = cv2.warpAffine(mask, inv_matrix, (background.shape[1], background.shape[0]), borderValue=0.0)

    _mask = mask.copy()
    _mask[_mask > 0.05] = 1.
    non_zero_points = cv2.findNonZero(_mask)
    _, _, w, h = cv2.boundingRect(non_zero_points)
    mask_size = int(np.sqrt(w * h))

    if erode_amount > 0:
        kernel_size = max(int(mask_size * erode_amount), 1)
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask = cv2.erode(mask, structuring_element)

    if blur_amount > 0:
        kernel_size = max(int(mask_size * blur_amount), 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    mask = np.tile(np.expand_dims(mask, axis=-1), (1, 1, 3))

    if blend_method == 'laplacian':
        composite_image = laplacian_blending(foreground, background, mask.clip(0,1), num_levels=4)
    else:
        composite_image = mask * foreground + (1 - mask) * background

    return composite_image.astype("uint8").clip(0, 255)


def image_mask_overlay(img, mask):
    img = img.astype('float32') / 255.
    img *= (mask + 0.25).clip(0, 1)
    img = np.clip(img * 255., 0., 255.).astype('uint8')
    return img


def resize_with_padding(img, expected_size=(640, 360), color=(0, 0, 0), max_flip=False):
    original_height, original_width = img.shape[:2]

    if max_flip and original_height > original_width:
        expected_size = (expected_size[1], expected_size[0])

    aspect_ratio = original_width / original_height
    new_width = expected_size[0]
    new_height = int(new_width / aspect_ratio)

    if new_height > expected_size[1]:
        new_height = expected_size[1]
        new_width = int(new_height * aspect_ratio)

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = cv2.copyMakeBorder(resized_img,
                                top=(expected_size[1] - new_height) // 2,
                                bottom=(expected_size[1] - new_height + 1) // 2,
                                left=(expected_size[0] - new_width) // 2,
                                right=(expected_size[0] - new_width + 1) // 2,
                                borderType=cv2.BORDER_CONSTANT, value=color)
    return canvas


def create_image_grid(images, size=128):
    num_images = len(images)
    num_cols = int(np.ceil(np.sqrt(num_images)))
    num_rows = int(np.ceil(num_images / num_cols))
    grid = np.zeros((num_rows * size, num_cols * size, 3), dtype=np.uint8)

    for i, image in enumerate(images):
        row_idx = (i // num_cols) * size
        col_idx = (i % num_cols) * size
        image = cv2.resize(image.copy(), (size,size))
        if image.dtype != np.uint8:
            image = (image.astype('float32') * 255).astype('uint8')
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        grid[row_idx:row_idx + size, col_idx:col_idx + size] = image

    return grid


def image_to_html(img, size=(640, 360), extension="jpg"):
    if img is not None:
        img = resize_with_padding(img, expected_size=size)
        buffer = cv2.imencode(f".{extension}", img)[1]
        base64_data = base64.b64encode(buffer.tobytes())
        imgbs64 = f"data:image/{extension};base64," + base64_data.decode("utf-8")
        html = '<div style="display: flex; justify-content: center; align-items: center; width: 100%;">'
        html += f'<img src={imgbs64} alt="No Preview" style="max-width: 100%; max-height: 100%;">'
        html += '</div>'
        return html
    return None


def mix_two_image(a, b, opacity=1.):
    a_dtype = a.dtype
    b_dtype = b.dtype
    a = a.astype('float32')
    b = b.astype('float32')
    a = cv2.resize(a, (b.shape[0], b.shape[1]))
    opacity = min(max(opacity, 0.), 1.)
    mixed_img = opacity * b + (1 - opacity) * a
    return mixed_img.astype(a_dtype)

resolution_map = {
        "Original": None,
        "240p": (426, 240),
        "360p": (640, 360),
        "480p": (854, 480),
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "1440p": (2560, 1440),
        "2160p": (3840, 2160),
    }

def resize_image_by_resolution(img, quality):
    resolution = resolution_map.get(quality, None)
    if resolution is None:
        return img

    h, w = img.shape[:2]
    if h > w:
        ratio = resolution[0] / h
    else:
        ratio = resolution[0] / w

    new_h, new_w = int(h * ratio), int(w * ratio)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def fast_pil_encode(pil_image):
    image_arr = np.asarray(pil_image)[:,:,::-1]
    buffer = cv2.imencode('.jpg', image_arr)[1]
    base64_data = base64.b64encode(buffer.tobytes())
    return "data:image/jpg;base64," + base64_data.decode("utf-8")

def fast_numpy_encode(img_array):
    buffer = cv2.imencode('.jpg', img_array)[1]
    base64_data = base64.b64encode(buffer.tobytes())
    return "data:image/jpg;base64," + base64_data.decode("utf-8")

crf_quality_by_resolution = {
    240: {"poor": 45, "low": 35, "medium": 28, "high": 23, "best": 20},
    360: {"poor": 35, "low": 28, "medium": 23, "high": 20, "best": 18},
    480: {"poor": 28, "low": 23, "medium": 20, "high": 18, "best": 16},
    720: {"poor": 23, "low": 20, "medium": 18, "high": 16, "best": 14},
    1080: {"poor": 20, "low": 18, "medium": 16, "high": 14, "best": 12},
    1440: {"poor": 18, "low": 16, "medium": 14, "high": 12, "best": 10},
    2160: {"poor": 16, "low": 14, "medium": 12, "high": 10, "best": 8}
}

def get_crf_for_resolution(resolution, quality):
    available_resolutions = list(crf_quality_by_resolution.keys())
    closest_resolution = min(available_resolutions, key=lambda x: abs(x - resolution))
    return crf_quality_by_resolution[closest_resolution][quality]