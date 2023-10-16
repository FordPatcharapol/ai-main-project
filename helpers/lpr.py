import re
import pytesseract
import easyocr
from helpers.logger import get_logger

from difflib import SequenceMatcher

# logger
logging = get_logger(__name__)

# reader = easyocr.Reader(['th'])
reader = easyocr.Reader(['th'], gpu=True)

countrys = [
    'กรุงเทพมหานคร', 'กระบี่', 'กาญจนบุรี', 'กาฬสินธุ์', 'กำแพงเพชร', 'ขอนแก่น', 'จันทบุรี', 'ฉะเชิงเทรา',
    'ชลบุรี', 'ชัยนาท', 'ชัยภูมิ', 'ชุมพร', 'เชียงราย', 'เชียงใหม่', 'ตรัง', 'ตราด', 'ตาก', 'นครนายก',
    'นครปฐม', 'นครพนม', 'นครราชสีมา', 'นครศรีธรรมราช', 'นครสวรรค์', 'นนทบุรี', 'นราธิวาส', 'น่าน', 'บึงกาฬ',
    'บุรีรัมย์', 'ปทุมธานี', 'ประจวบคีรีขันธ์', 'ปราจีนบุรี', 'ปัตตานี', 'พระนครศรีอยุธยา', 'พังงา',
    'พัทลุง', 'พิจิตร', 'พิษณุโลก', 'เพชรบุรี', 'เพชรบูรณ์', 'แพร่', 'พะเยา', 'ภูเก็ต', 'มหาสารคาม',
    'มุกดาหาร', 'แม่ฮ่องสอน', 'ยโสธร', 'ยะลา', 'ร้อยเอ็ด', 'ระนอง', 'ระยอง', 'ราชบุรี', 'ลพบุรี',
    'ลำปาง', 'ลำพูน', 'เลย', 'ศรีสะเกษ', 'สกลนคร', 'สงขลา', 'สตูล', 'สมุทรปราการ', 'สมุทรสงคราม',
    'สมุทรสาคร', 'สระแก้ว', 'สระบุรี', 'สิงห์บุรี', 'สุโขทัย', 'สุพรรณบุรี', 'สุราษฎร์ธานี', 'สุรินทร์',
    'หนองคาย', 'หนองบัวลำภู', 'อ่างทอง', 'อุดรธานี', 'อุตรดิตถ์', 'อุทัยธานี', 'อุบลราชธานี', 'อำนาจเจริญ'
]


pattern = re.compile(r"[^ก-๙0-9' ]|^'|'$|''")
pattern_plate = re.compile(r"[0-9]?[ก-ฮ]{2}[0-9]{2,4}")


def lpr_ocr_easy_ocr(image):
    results = reader.readtext(image, detail=1, paragraph=False)

    result_county = '-'
    conf_county = 0

    if results:
        data_txt = ''
        for txt in results:
            data_txt += txt[1]

        data_txt = data_txt.replace(' ', '')
        char_to_remove = re.findall(pattern, data_txt)
        list_with_char_removed = [
            char for char in data_txt if not char in char_to_remove]
        result_string = ''.join(list_with_char_removed)

        plate = re.findall(pattern_plate, result_string)

        if not plate:
            return None, None

        county = re.sub(plate[0], '', result_string)

        for idx_country in countrys:
            tmp = SequenceMatcher(None, idx_country, county).ratio()

            if tmp > conf_county:
                conf_county = tmp

                if conf_county > 0.65:
                    result_county = idx_country
                else:
                    result_county = '-'

        logging.debug('LPR_EASYOCR Extracting...')
        return plate, result_county

    return None, None


def lpr_ocr_tesseract(LP_image):
    for num in ["11", "12", "13"]:
        data_txt = pytesseract.image_to_string(
            LP_image, lang='tha', config='--dpi 2400 --oem 1 --psm '+num)

        char_to_remove = re.findall(pattern, data_txt)
        list_with_char_removed = [
            char for char in data_txt if not char in char_to_remove]
        result_string = ''.join(list_with_char_removed)

        plate = re.findall(pattern_plate, result_string)

        if not plate:
            continue

        country = re.sub(plate[0], '', result_string)

        logging.info('LPR_Tesseract Extracting... %s', num)
        return plate, country_mapper(country)

    return None, None


def country_mapper(country):
    conf_country = 0
    result_country = '-'

    for idx_country in countrys:
        tmp = SequenceMatcher(None, idx_country, country).ratio()

        if tmp > conf_country:
            conf_country = tmp

            if conf_country > 0.65:
                result_country = idx_country
            else:
                result_country = '-'

    return result_country
