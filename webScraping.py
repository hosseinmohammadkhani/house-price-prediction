import pandas as pd
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
from unidecode import unidecode
import os
import sys
import time
import tkinter as tk
from tkinter import ttk

# Get root path of the app
application_path = os.getcwd()

now = datetime.now()
today_date = now.strftime('%m%d%Y') # MMDDYYYY

service = Service(executable_path='./chromedriver.exe')
driver = webdriver.Chrome(service=service)

# containers' xpath
single_container = "//div[@class='YPVjY gYBq9 _6ySN8 GwUtl']/section/div[@class='SMz-b']/div[@class='a3w1n']"
end_of_the_list_container = "//div[@class='teSq5 oZa63']"

def create_dataset(area):

    collected_containers = []
    collected_links = []

    english_area = ""
    if area == "پونک" : english_area = "punak"
    elif area == "جنت آباد جنوبی" : english_area = "south-jannat-abad"
    elif area == "قیطریه" : english_area = "qeytarieh"
    elif area == "بلوار فردوس غرب" : english_area = "west-ferdows-boulevard"

    url = f"https://www.sheypoor.com/s/tehran/{english_area}/houses-apartments-for-sale"
    driver.get(url)

    while len(collected_containers) <= 5 : 
        containers = driver.find_elements(By.XPATH, single_container)
        for i in containers:
            if i not in collected_containers:
                collected_containers.append(i)
                link = i.find_element(By.XPATH , "./h2/a").get_attribute("href")
                collected_links.append(link)
                
        try:
            # Check if the end of the list is in view
            end_of_the_list = driver.find_element(By.XPATH, end_of_the_list_container)
            driver.execute_script("arguments[0].scrollIntoView();", end_of_the_list)
            break
        except:
            # Scroll down by 425px
            driver.execute_script("window.scrollBy(0, 425);")
            time.sleep(0.35)  # Adjust sleep time as needed

    # lists of features
    meterage_list = []
    rooms_number_list = []
    parkings_list = []
    storage_rooms_list = []
    elevators_list = []
    years_list = []
    prices_list = []

    print("Number of links collected : " , len(collected_links))

    for i in range(len(collected_links)) : 
        driver.get(collected_links[i])
        time.sleep(1.3)
        
        # If city of advertisement is different with city of user , then skip the advertisement
        city_in_webpage = ""
        try:
            city_in_webpage = driver.find_element(By.XPATH , "//div[@class='VEVgN']/nav/ul/li[1]/a").text
        except : 
            continue
        if city_in_webpage != "تهران" and city_in_webpage != "استان تهران" : 
            continue

        # Extract area of house. If it's equal to user's choice , extract the features else skip
        area_in_webpage = ""
        try : 
            area_in_webpage = driver.find_element(By.XPATH , "//div[@class='VEVgN']/nav/ul/li[5]/a").text
        except : 
            continue
        if area_in_webpage != area : 
            continue

        # Collect features from webpage and store it in features list
        features = driver.find_elements(By.XPATH, "//div[@class='C7Rh9']/p[2]")

        # Extract the price of house. If it's null , then skip it
        price = ""
        try : 
            price = driver.find_element(By.XPATH, "//span[@class='l29r1']/strong").text
            price_without_commas = price.replace("," , "")
            english_price_without_commas = unidecode(price_without_commas)
            price = int(english_price_without_commas)
        except : 
            print("No price")

        # If house has all excepted features , extract all of them
        # Else skip the house
        if(len(features) >= 7 and price != ""):
            meterage = features[0].text
            meterage = meterage.replace("," , "")
            english_meterage = int(unidecode(meterage))
            meterage_list.append(english_meterage)

            rooms_number = features[2].text
            if(rooms_number == "ندارد"):
                rooms_number = 0
            if(len(rooms_number) > 1):
                matches = re.findall(r'\d+', rooms_number)
                if matches:
                    number = int(matches[0])  # Convert the matched string to an integer
                    rooms_number = int(number)        
            else:
                rooms_number = int(unidecode(rooms_number))
            rooms_number_list.append(rooms_number)

            parking_check = True
            if features[3].text == "دارد" : 
                parking_check = True
            elif features[3].text == "ندارد":
                parking_check = False
            parkings_list.append(parking_check)

            storage_room_check = True
            if features[4].text == "دارد" : 
                storage_room_check = True
            elif features[4].text == "ندارد":
                storage_room_check = False
            storage_rooms_list.append(storage_room_check)
            
            elevator_check = True
            if features[5].text == "دارد" : 
                elevator_check = True
            elif features[5].text == "ندارد":
                elevator_check = False
            
            elevators_list.append(elevator_check)

            year = features[6].text
            matches = re.findall(r'[۰-۹]+', year)
            if matches:
                number = matches[0]
                year = int(unidecode(number))
            if features[6].text == "نوساز" : 
                year = 1
            years_list.append(year)

            prices_list.append(price)
            print("Information collected successfully")
        

    # Create csv file
    pdDict = {'Meterage' : meterage_list , 'Rooms' : rooms_number_list , 
            'Parkings' : parkings_list , 'Storage rooms' : storage_rooms_list,
            'Elevator' : elevators_list , 'Year' : years_list,
            'Price' : prices_list}
    csv_file = pd.DataFrame(pdDict)
    file_name = f'./Datasets/{english_area}-{today_date}.csv'
    file_path = os.path.join(application_path , file_name)
    csv_file.to_csv(file_path)

    driver.quit()

def submit_form():

    # Get name of the area from user
    area_value = area_var.get()
    
    # Create dataset
    create_dataset(area_value)


# Create the main application window
root = tk.Tk()
root.title("Property Information Form")

# Set the style
style = ttk.Style()
style.configure("TFrame", background="#f0f0f0")
style.configure("TLabel", background="#f0f0f0", font=("Arial", 12), anchor="e")
style.configure("TCombobox", font=("Arial", 12), justify="right")
style.configure("TButton", font=("Arial", 12), background="#4CAF50", foreground="black")
style.configure("TText", font=("Arial", 12))

# Define the main frame
main_frame = ttk.Frame(root, padding="20 20 20 20")
main_frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Define a variable for the Area field
area_var = tk.StringVar()

# Options for the Area combobox
area_options = ["پونک" , "جنت آباد جنوبی" , "قیطریه" , "بلوار فردوس غرب"]

# Define a uniform width for the combobox
uniform_width = 20

# Create and place the Area label and combobox
ttk.Label(main_frame, text="نام محله").grid(column=1, row=0, sticky=tk.W, padx=10, pady=10)
area_combobox = ttk.Combobox(main_frame, textvariable=area_var, justify="right", width=uniform_width)
area_combobox['values'] = area_options
area_combobox.grid(column=0, row=0, padx=10, pady=10, sticky=tk.E)

# Create a submit button
submit_button = ttk.Button(main_frame, text="Submit", style="TButton", command=submit_form)
submit_button.grid(column=0, row=1, columnspan=2, pady=20)

# Output section
output_label = ttk.Label(main_frame, text="Output:")
output_label.grid(column=0, row=2, pady=(20, 10), sticky=tk.W)

output_text = tk.Text(main_frame, width=50, height=10)
output_text.grid(column=0, row=3, columnspan=2, padx=10, pady=(0, 20))

# Move the window a little to the right
root.update_idletasks()
width = root.winfo_width()
height = root.winfo_height()
x = 100  # Adjust this value to move the window to the right
y = (root.winfo_screenheight() // 2) - (height // 2)
root.geometry(f'{width}x{height}+{x}+{y}')

# Run the application
root.mainloop()