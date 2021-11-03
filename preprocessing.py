import pandas as pd
import numpy as np
import argparse

def file2csv(file_name):
    data = []
    keys = ['age', 'type_employer', 'fnlwgt', 'education', 'education_num', 'marital',
            'occupation', 'relationship', 'race', 'sex', ' capital_gain', 'capital_loss',
            'hr_per_week', 'country', 'income']
    with open(file_name) as f:
        for line in f:
            attributes = line.strip().split(',')
            data_point = {}
            for idx, attribute in enumerate(attributes):
                data_point[keys[idx]] = np.nan if len(attribute.strip()) == 0 else attribute.strip()
            data.append(data_point)

    df = pd.DataFrame(data)
    # write to csv
    return df

def preprocessing(dataframe, datareadyfile):
    # to do
    # remove number with missing value
    dataframe.dropna(inplace=True)
    # remove attributes fnlwgt, education-num, relationship drop col
    filtered_df = dataframe(['fnlwgt', 'education-num', 'relationship'], axis=1)
    # binarize the following attributes:
    filtered_df[filtered_df['capital-gain'] > 0]['capital-gain'] = 'yes'  # yes > 0, no = 0
    filtered_df[filtered_df['capital-gain'] == 0]['capital-gain'] = 'no'
    filtered_df[filtered_df['capital-loss'] > 0]['capital-loss'] = 'yes'  # yes > 0, no = 0
    filtered_df[filtered_df['capital-loss'] == 0]['capital-loss'] = 'no'
    filtered_df[filtered_df['country'] != 'United-States']['country'] = 'other'  # United-States, other
    # discretize continuous attributes:
    category_age = ['young', 'adult', 'senior', 'old']
    filtered_df['age'] = pd.cut(x=filtered_df['age'],
                                bins=[0, 25, 45, 65, 90],
                                labels=category_age)
    category_hour = ['part-time', 'full-time', 'over-time']
    filtered_df['hr_per_week'] = pd.cut(x=filtered_df['hr_per_week'],
                                        bins=[0, 39, 40],
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
                                                                 'Some-college'])
    filtered_df['education'] = filtered_df['education'].replace(['Masters,'
                                                                 'Doctorate'],
                                                                'Grd')
    # martial status
    filtered_df['martial'] = filtered_df['martial'].replace(['Married-AF-spouse',
                                                             'Married-civ-spouse'],
                                                            'Married')

    filtered_df['martial'] = filtered_df['martial'].replace(['Married-spouse-absent',
                                                             'Separated',
                                                             'Divorced',
                                                             'Widowed'],
                                                            'Not-married')

    filtered_df['occupation'] = filtered_df['occupation'].replace(['Tech-support',
                                                                   'Adm-clerical',
                                                                   'Priv-house-serv',
                                                                   'Protective-serv',
                                                                   'Armed-Forces',
                                                                   'Other-service'],
                                                                  'Other')

    filtered_df['occupation'] = filtered_df['occupation'].replace(['Craft-repair', 'Farming-fishing',
                                                                   'Handlers-cleaners',
                                                                   'Machine-op-inspct',
                                                                   'Transport-moving'],
                                                                  'ManualWork')

    filtered_df.to_csv(datareadyfile, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help="Name to input file")
    parser.add_argument('--output_file', type=str, default="train.csv.", help="Name of csv output file")
    args = parser.parse_args()

    df = file2csv(args.input_file)
    preprocessing(df, args.output_file)
