# -*- coding: utf-8 -*-
"""
Created on Sun May 12 12:16:02 2024

@author: Lenovo
"""

# required inports 
import random
from collections import Counter


"""Code Started ----------Exercise 1: Prime Numbers------------------------
---------------------------------------------------------------------------"""
def number_is_prime_or_not():
    if num > 1:
        for i in range(2,num):
            if(num % i == 0):
                print(num, " is not a prime number")
                
            else:
                print(num, " is a prime number")
            break
    else:
        print(num, " is not a prime number")      
        
"""Code Started ----------Exercise 1: Prime Numbers--------------------------
---------------------------------------------------------------------------"""

"""Code Started ----------Exercise 2: Product of Random Numbers------------------------
---------------------------------------------------------------------------"""



def check_if_user_knows_the_product_of_random_number():
    random_number_generated = random.randrange(5, 10000, 3)
    print(random_number_generated)
    
    Number_1 = int(input("Can you please add your number 1: "))
    Number_2 = int(input("Can you please add your number 2: "))
    
    if((Number_1 * Number_2) == random_number_generated):
        print("Great Job! you answered correctly")
    else:
        print("Oops! you want to try again?")
        Question = input("If Yes, Please enter Yes else No.")
        Question = Question.lower()
        if(Question == "yes" or Question == "YES"):
            Number_1 = int(input("Can you please add your number 1: "))
            Number_2 = int(input("Can you please add your number 2: "))
            if((Number_1 * Number_2) == random_number_generated):
                    print("Great Job! you answered correctly")
            else:
                    print("Oops! hard luck next time!")
        else:
                print("Thanks for your time!")
    
"""Code Ended ----------Exercise 2: Product of Random Numbers--------------------------
---------------------------------------------------------------------------"""

"""Code Started ----------Exercise 3: Squares of Even/Odd Numbers--------------------------
---------------------------------------------------------------------------"""
def square_of_even_odd_numbers():
    for i in range(100,201,2):
        print(i*i)
    print("-------------------------------------------------------------")
    
    for i in range(101,200,2):
        print(i*i)    

"""Code Ended ----------Exercise 3: Squares of Even/Odd Numbers--------------------------
---------------------------------------------------------------------------"""

"""Code Started ----------Exercise 4: program to count the number of words in a given text--------------------------
---------------------------------------------------------------------------"""


        
def count_numbers_in_text():
    input_text = "This is a sample text. This text will be used to demonstrate the word counter."
    words = input_text.split()
    
    word_counts = Counter(words)
    
    for word,count in word_counts.items():
        print(f"{word}: {count}")
        
"""Code Ended ----------Exercise 4: program to count the number of words in a given text--------------------------
---------------------------------------------------------------------------"""

"""Code Started ----------Exercise 5: Check for Palindrome--------------------------
---------------------------------------------------------------------------"""
def rev(num):
    res = str(num) == str(num)[::-1]
    print("Is the number palindrome ? : " + str(res))
    
    return str(res)


"""Code Ended ----------Exercise 5: Check for Palindrome--------------------------
---------------------------------------------------------------------------"""





if __name__ =="__main__":
    print("""Exercise 1: Prime Numbers
              Write a Python program that checks whether a given number is prime or not. 
              A prime number is a natural number greater than 1 that has no positive
              divisors other than 1 and itself
              2, 3, 5, 7, 11, 13.""")

    num = int(input("Please enter a number to check if its prime or not: "))    
    number_is_prime_or_not()    
    
    print("""Exercise 2: Product of Random Numbers
              Develop a Python program that generates two random numbers and asks 
              the user to enter the product of these numbers. 
              The program should then check if the user's answer is correct and 
              display an appropriate message.""")

    check_if_user_knows_the_product_of_random_number() 
    
    
    print("""Exercise 3: Squares of Even/Odd Numbers
          Create a Python script that prints the squares of all even or odd numbers within the range of 100 to 200.
          Choose either even or odd numbers and document your choice in the code.""")
    square_of_even_odd_numbers()   
    
    print("""write a program to count the number of words in a given text.
            example:
            input_text = "This is a sample text. This text will be used to demonstrate the word counter."
            Expected output:
            'This': 2 
            'is': 1
            'a': 1
            'sample': 1
            'text.': 1
            """)
    count_numbers_in_text()         
    print("""Exercise 5: Check for Palindrome
            Write a Python function called is_palindrome that takes a string as input and returns True if the string is a palindrome, and False otherwise. A palindrome is a word, phrase, number, or other sequence of characters that reads the same forward and backward, ignoring spaces, punctuation, and capitalization.
            Example:
            Input: "racecar"
            Expected Output: True
            """)
    rev("rahulram")        
