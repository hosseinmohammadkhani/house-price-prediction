import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import explained_variance_score
import tkinter as tk
from tkinter import ttk

def submit_form_and_print_output():

    # Function to handle form submission
    area_value = fields["نام محله"].get()
    meterage_value = fields["متراژ"].get()
    number_of_rooms_value = fields["تعداد اتاق"].get()
    parking_value = fields["پارکینگ"].get()
    storage_room_value = fields["انباری"].get()
    elevator_value = fields["آسانسور"].get()
    year_value = fields["سال ساخت"].get()
    algorithm_value = fields["الگوریتم رگرسیون"].get()

    csv_name = ""
    if area_value == "جنت آباد جنوبی" : 
        csv_name = "./Datasets/jannat-abad.csv"
    elif area_value == "پونک":
        csv_name = "./Datasets/punak.csv"
    elif area_value == "بلوار فردوس غرب":
        csv_name = "./Datasets/bolvare-ferdose-qarb.csv"
    elif area_value == "قیطریه" : 
        csv_name = "./Datasets/qeytarieh.csv"


    # Read csv file 
    df = pd.read_csv(csv_name)

    # Features and label variable
    x = df[['Meterage', 'Rooms' , 'Parkings' , 'Storage rooms' , 'Elevator' , 'Year']]
    y = df['Price']

    # 80% of data for training and 20% for testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

    # Standardize the features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    def evaluate_model(model , x_test , y_test):
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test , y_pred)
        rmse = np.sqrt(mse)
        variance = "{0:.2f}".format(explained_variance_score(y_test , y_pred))
        output_text.insert(tk.END, f"Variance score : {variance} \n")
        output_text.insert(tk.END, f"RMSE : {('{:,}'.format(int(rmse)))} Tooman\n")

        plt.scatter(y_test, y_pred, color='blue')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.title('Real vs. Predicted Prices')
        plt.xlabel('Real Price')
        plt.ylabel('Predicted Price')
        plt.show()
    
    def knn_algorithm(meterage , rooms , parkings , storage_rooms , elevator , year):

        # Train the KNN regressor
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(x_train, y_train)

        # User's input
        user_input = {
            'Meterage': meterage,
            'Rooms': rooms,
            'Parkings': parkings,
            'Storage rooms': storage_rooms,
            'Elevator': elevator,
            'Year': year
        }

        # Convert the new data point to a DataFrame
        user_input_df = pd.DataFrame([user_input])

        # Standardize the user's input
        standard_user_input_df = scaler.transform(user_input_df)

        # Predict the price
        predicted_price = knn.predict(standard_user_input_df)

        # Print the predicted price
        predicted_price = ('{:,}'.format(int(predicted_price[0])))
        output_text.insert(tk.END, f"Predicted Price: {predicted_price} Tooman\n")

        # Evaluate the model
        evaluate_model(knn , x_test , y_test)


    def linear_regression_algorithm(meterage , rooms , parkings , storage_rooms , elevator , year):

        # Train the model using 80% of data
        linear_regression = LinearRegression()
        linear_regression.fit(x_train , y_train)

        # User's input
        user_input = {
            'Meterage': meterage,
            'Rooms': rooms,
            'Parkings': parkings,
            'Storage rooms': storage_rooms,
            'Elevator': elevator,
            'Year': year
        }

        # Convert the new data point to a DataFrame
        user_input_df = pd.DataFrame([user_input])

        # Standardize the user's input
        standard_user_input_df = scaler.transform(user_input_df)

        # Predict the price
        predicted_price = linear_regression.predict(standard_user_input_df)

        # Print the predicted price
        predicted_price = ('{:,}'.format(int(predicted_price[0])))
        output_text.insert(tk.END, f"Predicted Price: {predicted_price} Tooman\n")

        # Evaluate the model
        evaluate_model(linear_regression , x_test , y_test)


    def gaussian_NB(meterage , rooms , parkings , storage_rooms , elevator , year):

        nb_model = GaussianNB()
        nb_model.fit(x_train , y_train)
        
        # User's input
        user_input = {
            'Meterage': meterage,
            'Rooms': rooms,
            'Parkings': parkings,
            'Storage rooms': storage_rooms,
            'Elevator': elevator,
            'Year': year
        }

        # Convert the new data point to a DataFrame
        user_input_df = pd.DataFrame([user_input])

        # Standardize the user's input
        standard_user_input_df = scaler.transform(user_input_df)

        # Predict the price
        predicted_price = nb_model.predict(standard_user_input_df)

        # Print the predicted price
        predicted_price = ('{:,}'.format(int(predicted_price[0])))
        output_text.insert(tk.END, f"Predicted Price: {predicted_price} Tooman\n")

        # Evaluate the model
        evaluate_model(nb_model , x_test , y_test)

    def decision_tree_algorithm(meterage , rooms , parkings , storage_rooms , elevator , year):

        tr_regressor = DecisionTreeRegressor(random_state=50)
        tr_regressor.fit(x_train,y_train)

        # User's input
        user_input = {
            'Meterage': meterage,
            'Rooms': rooms,
            'Parkings': parkings,
            'Storage rooms': storage_rooms,
            'Elevator': elevator,
            'Year': year
        }

        # Convert the new data point to a DataFrame
        user_input_df = pd.DataFrame([user_input])

        # Standardize the user's input
        standard_user_input_df = scaler.transform(user_input_df)

        # Predict the price
        predicted_price = tr_regressor.predict(standard_user_input_df)

        # Print the predicted price
        predicted_price = ('{:,}'.format(int(predicted_price[0])))
        output_text.insert(tk.END, f"Predicted Price: {predicted_price} Tooman\n")

        # Evaluate the model
        evaluate_model(tr_regressor , x_test , y_test)

    parking_value = True if parking_value == "دارد" else False
    storage_room_value = True if storage_room_value == "دارد" else False
    elevator_value = True if elevator_value == "دارد" else False

    year_value = 1403 - int(year_value)
    if year_value == 0 : year_value = 1

    if algorithm_value == "KNN":
        knn_algorithm(meterage_value , number_of_rooms_value , parking_value , storage_room_value , elevator_value , year_value)
    if algorithm_value == "Linear regression":
        linear_regression_algorithm(meterage_value , number_of_rooms_value , parking_value , storage_room_value , elevator_value , year_value)
    if algorithm_value == "Gaussian naive bayes" : 
        gaussian_NB(meterage_value , number_of_rooms_value , parking_value , storage_room_value , elevator_value , year_value)
    if algorithm_value == "Decision tree" : 
        decision_tree_algorithm(meterage_value , number_of_rooms_value , parking_value , storage_room_value , elevator_value , year_value)

    # Clear previous content
    output_text.delete(1.0, tk.END)

    # Display new content
    '''
    output_text.insert(tk.END, f"Area: {area_value}\n")
    output_text.insert(tk.END, f"Meterage: {meterage_value}\n")
    output_text.insert(tk.END, f"Number of rooms: {num_rooms_value}\n")
    output_text.insert(tk.END, f"Parking: {parking_value}\n")
    output_text.insert(tk.END, f"Storage room: {storage_room_value}\n")
    output_text.insert(tk.END, f"Elevator: {elevator_value}\n")
    output_text.insert(tk.END, f"Year: {year_value}\n")
    '''

# Create the main application window
root = tk.Tk()
root.title("Property Information Form")

# Set the style
style = ttk.Style()
style.configure("TFrame", background="#f0f0f0")
style.configure("TLabel", background="#f0f0f0", font=("Arial", 12), anchor="e")
style.configure("TEntry", font=("Arial", 12), justify="right")
style.configure("TCombobox", font=("Arial", 12), justify="right")
style.configure("TButton", font=("Arial", 12), background="#4CAF50", foreground="black")
style.configure("TText", font=("Arial", 12))

# Define the main frame
main_frame = ttk.Frame(root, padding="20 20 20 20")
main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Define a dictionary for labels and their corresponding variables
fields = {
    "نام محله": tk.StringVar(),
    "متراژ": tk.StringVar(),
    "تعداد اتاق": tk.StringVar(),
    "پارکینگ": tk.StringVar(),
    "انباری": tk.StringVar(),
    "آسانسور": tk.StringVar(),
    "سال ساخت": tk.StringVar(),
    "الگوریتم رگرسیون": tk.StringVar()
}

# Options for comboboxes
area_options = ["پونک", "بلوار فردوس غرب" , "جنت آباد جنوبی" , "قیطریه"]
parking_options = ["دارد" , "ندارد"]
storage_room_options = ["دارد" , "ندارد"]
elevator_options = ["دارد" , "ندارد"]
algorithm_options = ["KNN" , "Linear regression" , "Gaussian naive bayes" , "Decision tree"]

# Define a smaller width for both entry and combobox
uniform_width = 20

# Create and place the labels and entry widgets
for idx, (label, var) in enumerate(fields.items()):
    if label in ["نام محله", "پارکینگ", "انباری", "آسانسور" , "الگوریتم رگرسیون"]:
        combobox = ttk.Combobox(main_frame, textvariable=var, justify="right", width=uniform_width)
        if label == "نام محله":
            combobox['values'] = area_options
        elif label == "پارکینگ":
            combobox['values'] = parking_options
        elif label == "انباری":
            combobox['values'] = storage_room_options
        elif label == "آسانسور":
            combobox['values'] = elevator_options
        elif label == "الگوریتم رگرسیون":
            combobox['values'] = algorithm_options
        combobox.grid(column=0, row=idx, padx=10, pady=10, sticky=tk.E)
    else:
        ttk.Entry(main_frame, width=uniform_width, textvariable=var, justify="right").grid(column=0, row=idx, padx=10, pady=10, sticky=tk.E)
    ttk.Label(main_frame, text=label).grid(column=1, row=idx, sticky=tk.W, padx=10, pady=10)

# Add some padding around all the child widgets of main_frame
for child in main_frame.winfo_children():
    child.grid_configure(padx=10, pady=10)

# Create a submit button
submit_button = ttk.Button(main_frame, text="پیش بینی قیمت", style="TButton", command=submit_form_and_print_output)
submit_button.grid(column=0, row=len(fields), columnspan=2, pady=20)

# Output section
output_label = ttk.Label(main_frame, text="Output:")
output_label.grid(column=0, row=len(fields)+1, pady=(20, 10), sticky=tk.W)

output_text = tk.Text(main_frame, width=50, height=10)
output_text.grid(column=0, row=len(fields)+2, columnspan=2, padx=10, pady=(0, 20))

# Center the window on the screen
root.update_idletasks()
width = root.winfo_width()
height = root.winfo_height()
x = 800
y = (root.winfo_screenheight() // 2) - (height // 2)
root.geometry(f'{width}x{height}+{x}+{y}')

# Run the application
root.mainloop()
