from mimic_iv_config import admission_data_path, patient_data_path
import csv
from itertools import islice
from datetime import datetime


def read_admission_data(path):
    data_dict = dict()
    with open(path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            subject_id, hadm_id, admit_time = line[:3]
            if subject_id not in data_dict:
                data_dict[subject_id] = dict()
            data_dict[subject_id][hadm_id] = admit_time
    return data_dict


def read_patient_data(path):
    sex_dict = dict()
    birth_year_dict = dict()
    with open(path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            subject_id, gender, age, year = line[:4]
            sex_dict[subject_id] = gender
            birth_year_dict[subject_id] = int(age), int(year)
    return sex_dict, birth_year_dict


def calculate_time_difference(year_str, date_str):
    # 将年份字符串转为 datetime 对象，假设每年的日期为年初的 1 月 1 日 00:00:00
    year_start = datetime(int(year_str), 1, 1, 0, 0, 0)

    # 将具体的日期字符串转为 datetime 对象
    specific_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

    # 计算时间差（以秒为单位）
    difference_in_seconds = (specific_date - year_start).total_seconds()

    # 每年的总秒数（考虑平年和闰年）
    seconds_in_year = 365 * 24 * 3600
    if (int(year_str) % 4 == 0 and int(year_str) % 100 != 0) or (int(year_str) % 400 == 0):
        seconds_in_year = 366 * 24 * 3600

    # 将时间差转换成年份的整数部分和小数部分
    difference_in_years = difference_in_seconds / seconds_in_year

    # 分离年份的整数部分和小数部分
    years_integer = int(difference_in_years)
    years_decimal = difference_in_years - years_integer

    # 返回结果，年份整数部分加上小数部分
    return years_integer + years_decimal


def main():
    visit_dict = read_admission_data(admission_data_path)
    sex_dict, birth_year_dict = read_patient_data(patient_data_path)

    sex_distribution = {}
    for key in sex_dict:
        sex = sex_dict[key]
        if sex not in sex_distribution:
            sex_distribution[sex] = 0
        sex_distribution[sex] += 1
    print(sex_distribution)

    age_dict = dict()
    for subject_id in visit_dict:
        if subject_id not in birth_year_dict:
            continue
        anchor_age, birth_year = birth_year_dict[subject_id]
        for visit_id in visit_dict[subject_id]:
            admit_time = visit_dict[subject_id][visit_id]
            offset = calculate_time_difference(birth_year, admit_time)
            age_dict[subject_id+'-'+visit_id] = offset + anchor_age
    age_distribution = {"18-40": 0, "40-60": 0, "60-80": 0, "80+": 0}
    for key in age_dict:
        age = age_dict[key]
        if age < 40:
            age_distribution["18-40"] += 1
        elif age < 60:
            age_distribution["40-60"] += 1
        elif age < 80:
            age_distribution["60-80"] += 1
        else:
            age_distribution["80+"] += 1
    print(age_distribution)


if __name__ == '__main__':
    main()
