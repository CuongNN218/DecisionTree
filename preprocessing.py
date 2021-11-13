import pandas as pd
import numpy as np
import argparse


def file2csv(file_name):
    data = []
    keys = ['age', 'type_employer', 'fnlwgt', 'education', 'education_num', 'marital',
            'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
            'hr_per_week', 'country', 'income']
    with open(file_name) as f:
        for line in f:
            attributes = line.strip().split(',')
            data_point = {}
            for idx, attribute in enumerate(attributes):
                data_point[keys[idx]] = np.nan if attribute.strip() == '?' else attribute.strip()
            data.append(data_point)

    df = pd.DataFrame(data)
    integer_attributes = ['age', 'hr_per_week', 'capital_gain', 'capital_loss']
    df[integer_attributes] = df[integer_attributes].astype(int)

    return df


def preprocessing(dataframe, data_ready_file):
    # to do
    # print(dataframe.shape)
    # remove number with missing value
    dataframe = dataframe.dropna(inplace=False)
    # print(dataframe.shape)
    # remove attributes fnlwgt, education-num, relationship drop col
    filtered_df = dataframe.drop(['fnlwgt', 'education_num', 'relationship'], axis=1)
    # binarize the following attributes:
    filtered_df.loc[filtered_df['capital_gain'] > 0, 'capital_gain'] = 'yes'  # yes > 0, no = 0
    filtered_df.loc[filtered_df['capital_gain'] == 0, 'capital_gain'] = 'no'
    filtered_df.loc[filtered_df['capital_loss'] > 0, 'capital_loss'] = 'yes'  # yes > 0, no = 0
    filtered_df.loc[filtered_df['capital_loss'] == 0, 'capital_loss'] = 'no'
    filtered_df.loc[filtered_df['country'] != 'United-States', 'country'] = 'other'  # United-States, other
    # discretize continuous attributes:
    category_age = ['young', 'adult', 'senior', 'old']
    filtered_df['age'] = pd.cut(x=filtered_df['age'],
                                bins=[0, 25, 45, 65, 90],
                                labels=category_age)
    category_hour = ['part-time', 'full-time', 'over-time']
    filtered_df['hr_per_week'] = pd.cut(x=filtered_df['hr_per_week'],
                                        bins=[0, 39, 40, 168],
                                        labels=category_hour)
    # merge attribute values together/reassign attribute values
    # work_class
    filtered_df['type_employer'] = filtered_df['type_employer'].replace(['Federal-gov',
                                                                         'Local-gov',
                                                                         'State-gov'], 'gov')
    filtered_df['type_employer'] = filtered_df['type_employer'].replace(['Without-pay',
                                                                         'Never-worked'],
                                                                        'Not-working')
    filtered_df['type_employer'] = filtered_df['type_employer'].replace(['Self-emp-inc',
                                                                         'Self-emp-not-inc'],
                                                                        'Self-employed')
    # education

    filtered_df['education'] = filtered_df['education'].replace(['Preschool',
                                                                 '1st-4th',
                                                                 '5th-6th',
                                                                 '7th-8th',
                                                                 '9th',
                                                                 '10th',
                                                                 '11th',
                                                                 '12th'],
                                                                'BeforeHS')

    filtered_df['education'] = filtered_df['education'].replace(['Prof-school',
                                                                 'Assoc-acdm',
                                                                 'Assoc-voc',
                                                                 'Some-college'],
                                                                'AfterHS')
    filtered_df['education'] = filtered_df['education'].replace(['Masters',
                                                                 'Doctorate'],
                                                                'Grd')
    # martial status
    filtered_df['marital'] = filtered_df['marital'].replace(['Married-AF-spouse',
                                                             'Married-civ-spouse'],
                                                            'Married')

    filtered_df['marital'] = filtered_df['marital'].replace(['Married-spouse-absent',
                                                             'Separated',
                                                             'Divorced',
                                                             'Widowed'],
                                                            'Not-married')
    # occupation
    filtered_df['occupation'] = filtered_df['occupation'].replace(['Tech-support',
                                                                   'Adm-clerical',
                                                                   'Priv-house-serv',
                                                                   'Protective-serv',
                                                                   'Armed-Forces',
                                                                   'Other-service'],
                                                                  'Other')

    filtered_df['occupation'] = filtered_df['occupation'].replace(['Craft-repair',
                                                                   'Farming-fishing',
                                                                   'Handlers-cleaners',
                                                                   'Machine-op-inspct',
                                                                   'Transport-moving'],
                                                                  'ManualWork')

    filtered_df.to_csv(data_ready_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help="Name to input file")
    parser.add_argument('--output_file', type=str, default="train.csv", help="Name of csv output file")
    args = parser.parse_args()

    df_adult = file2csv(args.input_file)
    preprocessing(df_adult, args.output_file)
