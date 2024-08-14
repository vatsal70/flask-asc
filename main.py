from process_df import preprocess_data
from train_model import model_train
from prediction_file import predict_sales
import os
import joblib



file_paths = ['FINAL/extracted/xgboost_model.pkl', 'FINAL/extracted/label_encoder_sales_office.pkl', 'FINAL/extracted/label_encoder_material.pkl', 'FINAL/extracted/label_encoder_sold_to_code.pkl']

file_exists = False
for item in file_paths:
    if os.path.exists(item):
        print(f"File {item} exists.")
        file_exists = True
    else:
        print("File does not exist.")
        file_exists = False


def get_input_from_options(options, type):
    """
    Prompt the user to select an input from a predefined set of options.

    Args:
        options (list): List of options from which the user can select.

    Returns:
        str: The selected option.
    """
    while True:
        print("Select an option:")
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        try:
            if(type == 'int'):
                choice = int(input("Enter the value from the option: "))
            else:
                choice = str(input("Enter the value from the option: "))
            if choice in options:
                return choice
        except ValueError:
            print("Invalid input! Please enter a number.")



label_encoder_sales_office, label_encoder_material, label_encoder_sold_to_code, df = preprocess_data()
model = joblib.load('FINAL/extracted/xgboost_model.pkl')

if(file_exists == False):
    model = model_train(label_encoder_sales_office, label_encoder_material, label_encoder_sold_to_code, df)
else:
    material_option = ['Regular PPC', 'Ambuja Plus', 'COMPOCEM', 'Kawach', 'OPC_53']
    sales_office_options = ['NP02', 'NP03', 'ZN8B', 'ZN8C', 'ZN8D', 'ZN8E', 'ZN8F', 'ZN8G', 'ZN8L']
    county_code_options = [265, 892, 839, 865, 893, 840, 888, 848, 833, 264, 845, 837, 829, 894, 831, 262, 844, 895, 838, 841, 885, 887, 886, 832, 889, 830, 846, 891, 842, 890, 849, 843, 836]

    
    sold_to_code = choice = input("Enter the dealer code: ")
    print(f"You selected: {sold_to_code}")

    material = get_input_from_options(material_option, 'str')
    print(f"You selected: {material}")

    sales_office = get_input_from_options(sales_office_options, 'str')
    print(f"You selected: {sales_office}")

    county_code = get_input_from_options(county_code_options, 'int')
    print(f"You selected: {county_code}")

    predict_sales(sold_to_code, material, sales_office, county_code, model, label_encoder_sold_to_code, label_encoder_material, label_encoder_sales_office, df)

# sold_to_code = '9121004029'
# material = 'Regular PPC'
# county_code = 841
# sales_office = 'ZN8E'