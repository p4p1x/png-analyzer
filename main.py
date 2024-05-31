import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import zlib

def display_fourier_transform_rgb(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_fourier_r = np.fft.fftshift(np.fft.fft2(img_rgb[:, :, 0]))
    img_fourier_g = np.fft.fftshift(np.fft.fft2(img_rgb[:, :, 1]))
    img_fourier_b = np.fft.fftshift(np.fft.fft2(img_rgb[:, :, 2]))

    fig, ax = plt.subplots(4, 3,
                           figsize=(18, 18))

    ax[0,1].imshow(img_rgb)
    ax[0,1].set_title('Original Image (RGB)')

    for i, (channel_name, img_fourier) in enumerate(
            [('Red', img_fourier_r), ('Green', img_fourier_g), ('Blue', img_fourier_b)]):
        magnitude_spectrum = np.log(np.abs(img_fourier) + 1)
        ax[1,0+i].imshow(magnitude_spectrum, cmap='gray')
        ax[1, 0+i].set_title(f'Fourier Transform (Magnitude) - {channel_name}')

    for i, (channel_name, img_fourier) in enumerate(
            [('Red', img_fourier_r), ('Green', img_fourier_g), ('Blue', img_fourier_b)]):
        phase_spectrum = np.angle(img_fourier)
        ax[2, 0+i].imshow(phase_spectrum, cmap='gray')
        ax[2, 0+i].set_title(f'Fourier Transform (Phase) - {channel_name}')

    # Inverse Fourier Transform for each color channel
    for i, (channel_name, img_fourier) in enumerate(
            [('Red', img_fourier_r), ('Green', img_fourier_g), ('Blue', img_fourier_b)]):
        inverse_fourier = np.abs(np.fft.ifft2(np.fft.ifftshift(img_fourier)))
        ax[3, 0+i].imshow(inverse_fourier, cmap='gray')
        ax[3, 0+i].set_title(f'Inverse Fourier Transform - {channel_name}')

    plt.show()


def decode_IHDR(chunk_data):
    width = int.from_bytes(chunk_data[:4], byteorder='big')
    height = int.from_bytes(chunk_data[4:8], byteorder='big')
    bit_depth = int.from_bytes(chunk_data[8:9], byteorder='big')
    color_type = int.from_bytes(chunk_data[9:10], byteorder='big')
    compression_method = int.from_bytes(chunk_data[10:11], byteorder='big')
    filter_method = int.from_bytes(chunk_data[11:12], byteorder='big')
    interlace_method = int.from_bytes(chunk_data[12:13], byteorder='big')
    print("width", width, "height", height, "bit_depth", bit_depth, "color_type", color_type, "compression", compression_method,
          "filter", filter_method, "interlace", interlace_method)

def decode_TEXT(chunk_data):
    # checking if works for more than one keyword in chunk
    #chunk_data = chunk_data + b'\x00' + chunk_data
    keyword_text: list = chunk_data.split(b'\x00')
    for index in range(0, len(keyword_text), 2):
        keyword = keyword_text[index].decode('utf-8')
        text = keyword_text[index + 1].decode('utf-8')
        print(f"Keyword: {keyword}, Text: {text}")

def decode_zTXt(chunk_data):
    keyword_end = chunk_data.find(b'\x00')
    keyword = chunk_data[:keyword_end].decode('utf-8') # b'\x00' - null byte (separator)
    compression_method = chunk_data[keyword_end+1]
    text_not_decompressed = chunk_data[keyword_end+2:]
    if compression_method == 0:
        text = zlib.decompress(text_not_decompressed)
    print(f"Keyword: {keyword}, Compression method: {compression_method}, Text: {text.decode()}")


def decode_iTXt(chunk_data):
    keyword_end = chunk_data.find(b'\x00')
    keyword = chunk_data[:keyword_end].decode('utf-8')  # b'\x00' - null byte (separator)
    compression_flag = chunk_data[keyword_end + 1]
    compression_method = chunk_data[keyword_end + 2]

    language_tag_start = keyword_end + 3
    language_tag_end = chunk_data.find(b'\x00', language_tag_start)
    language_tag = chunk_data[language_tag_start:language_tag_end].decode('utf-8')

    translated_keyword_start = language_tag_end + 1
    translated_keyword_end = chunk_data.find(b'\x00', translated_keyword_start)
    translated_keyword = chunk_data[translated_keyword_start:translated_keyword_end].decode('utf-8')

    text_start = translated_keyword_end + 1
    text = chunk_data[text_start:].decode('utf-8')

    print(f"Keyword: {keyword}")
    print(f"Compression Flag: {compression_flag}")
    print(f"Compression Method: {compression_method}")
    print(f"Language Tag: {language_tag}")
    print(f"Translated Keyword: {translated_keyword}")
    print(f"Text: {text}")


def decode_PLTE(chunk_data, chunk_length):
    """   Red:   1 byte (0 = black, 255 = red)
   Green: 1 byte (0 = black, 255 = green)
   Blue:  1 byte (0 = black, 255 = blue) """
    len = int.from_bytes(chunk_length)
    groups = 0
    while groups < len:
        red = chunk_data[groups]
        green = chunk_data[groups+1]
        blue = chunk_data[groups+2]
        groups += 3
        print(f"Palette entry no. {groups/3}, Red: {red}, Green: {green}, Blue: {blue}")

def decode_TIME(chunk_data):
    year = int.from_bytes(chunk_data[:2], byteorder='big')
    month = int.from_bytes(chunk_data[2:3], byteorder='big')
    day = int.from_bytes(chunk_data[3:4], byteorder='big')
    hour = int.from_bytes(chunk_data[4:5], byteorder='big')
    minute = int.from_bytes(chunk_data[5:6], byteorder='big')
    second = int.from_bytes(chunk_data[6:7], byteorder='big')
    print(f"Year: {year}, Month: {month}, Day: {day}, Hour: {hour}, Minute: {minute}, Second: {second}")


def chunks_splitter(file_path):
    chunks = []
    chunk_length = []
    chunk_type = []
    chunk_data = []
    chunk_crc = []
    header = []
    with open(file_path, 'rb') as file:
        head = file.read(8)
        header.append(head)
        print(head)
        aa = 'a'
        aprim = int(aa, 16)
        print(aprim)
        while True:
            chunk_l = file.read(4)
            chunk_length.append(chunk_l)
            chunk_t = file.read(4)
            chunk_type.append(chunk_t)
            if chunk_l == 0 or chunk_t == b'IEND':  # if we want to read file to the theoretical end
                length = int.from_bytes(chunk_l, byteorder='big')
                chunk_d = file.read(length)
                chunk_data.append(chunk_d)
                chunk_c = file.read(4)
                chunk_crc.append(chunk_c)
                chunks.append((chunk_l + chunk_t + chunk_d + chunk_c))
                break
            # if not chunk_l and chunk_t != b'IEND':  # if we want to read file to the real end
            #     break
            length = int.from_bytes(chunk_l, byteorder='big')
            chunk_d = file.read(length)
            chunk_data.append(chunk_d)
            chunk_c = file.read(4)
            chunk_crc.append(chunk_c)
            chunks.append((chunk_l+chunk_t+chunk_d+chunk_c))

    return chunks, chunk_length, chunk_type, chunk_crc, chunk_data

def print_chunks(chunks):
    for k in range(len(chunks)):
        print("Chunk Length:", int.from_bytes(chunk_length[k], byteorder='big'))
        print("Chunk Type:", chunk_type[k])
        if chunk_type[k] == b'IHDR':
            decode_IHDR(chunk_data[k])
        elif chunk_type[k] == b'tEXt':
            decode_TEXT(chunk_data[k])
        elif chunk_type[k] == b'zTXt':
            decode_zTXt(chunk_data[k])
        elif chunk_type[k] == b'iTXt':
            decode_iTXt(chunk_data[k])
        elif chunk_type[k] == b'PLTE':
            decode_PLTE(chunk_data[k], chunk_length[k])
        elif chunk_type[k] == b'tIME':
            decode_TIME(chunk_data[k])
        print("\n")
def chunks_merge(chunks):
    header = b'\x89PNG\r\n\x1a\n'
    chunk = b''
    for c in range(len(chunks)):
        chunk += chunks[c]
    merged_png = header + chunk
    return merged_png

def anonymize_chunks(chunks):
    critical_chunk_types = [b'IHDR', b'PLTE', b'IDAT', b'IEND']
    filtered_chunks = []
    iend_count = 0
    for chunk in chunks:
        chunk_type = chunk[4:8]
        if chunk_type in critical_chunk_types:
            if chunk_type == b'IEND' and iend_count == 1:
                continue
            elif chunk_type == b'IEND':
                iend_count += 1
            filtered_chunks.append(chunk)
    anonymized_png = chunks_merge(filtered_chunks)

    return anonymized_png

if __name__ == "__main__":
    file_path = "lena_rgb.png"
    display_fourier_transform_rgb(file_path)
    img = Image.open(file_path)
    chunks, chunk_length, chunk_type, chunk_crc, chunk_data = chunks_splitter(file_path)
    print_chunks(chunks)

    anonymized_png = anonymize_chunks(chunks)
    with open("anonymized.png", 'wb') as anonymized_file:
        anonymized_file.write(anonymized_png)
