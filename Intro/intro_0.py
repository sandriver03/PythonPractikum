# suppose we are creating a website, and need to ask the user to register an account
# when registering, each account should have the following information:
#   account name
#   user age
#   sex
# lets suppose we are making a health survey website, so we are also asking the following information:
#   height
#   weight

# now lets work on the registration process. a good way of thinking about solving problems with
# programing is to break down the problem into a series of steps to be solved sequentially
# for the registration, we just ask the new user for the needed information one by one
# to ask the user to type in something with keyboard, in Python you can use the 'input' function
# you can always check the documentation to see how to use each and every function/command
user_name = input('Please choose your username: \n')
user_age = input('Please tell us your age: \n')
user_sex = input('Please tell us your sex: \n')
user_height = input('Please tell us your height in cm: \n')
user_weight = input('Please tell us your weight in kg: \n')

# do you see any problems with the code above? what could go wrong?
# remember that the compiler, which is the part of a programming language that translate your code into
# machine instructions 'exactly' as you specified, you need to be very precise as what your instruction
# really does
# e.g. what if some one typed in some non-sense, like 41 in sex?
# do we need any restrictions on the user name?
# also, what are the data types you got for each question? do they make sense?
# you can use type casting to convert among different data types


# now we have the height and weight of the user, we can use it to calculate the Body Mass Index (BMI,
# see https://en.wikipedia.org/wiki/Body_mass_index)


# now, we know how to get one user. but for a website, I think more users are going to be much more
# helpful (and nicer). now let ask multiple people to register for their own unique accounts
# lets say, each user need to have an unique account name, treating upper case letters the same as
# lower case ones. i.e. 'Chao' is the same as 'CHAO' or 'chao', if 'Chao' is already chosen then
# 'CHAO' or 'chao' and so on cannot be chosen again
# how do you do this? think about all the things you need, and break down the problem in steps
# do you need to store all registered users somehow? how to you do this? what data type you want to
# use?


# now you can see that we are using the asking inputs lines multiple times. we can instead turn those
# lines into a function, which will save us a lot of typing times (yeh I know we are obsessed about
# this. I once had a conversation with a friend about why Python is much superior to Matlab because
# you use 'elif' instead of 'elseif' which saves you 2 characters, big deal!), and also later on
# whenever you need to do this again, you already have the tool at hand


# we should also make a function to calculate the BMI


# finally, saving and loading data
# if we save the user information we already have, then next time whenever we want to use it, we can
# directly load it into our program
# first decide if you want to use pickle or json
# you need to decide where you want to save the data, and what file name you want to choose
# if you do not specify an exact path, the file will be created in current working directory
# to get your current working directory, import the os module, and use os.getcwd()
# save data with pickle
import pickle
# path I want to save my file
file_path = 'c:\\users\\pc\\desktop\\temp'
file_name = 'PyIntro.pkl'
# now open the file and get the file handle
file_handle = open(file_path + '\\' + file_name, 'wb')
# save variable var_name into the pickle file (var_name should be the variable you want to save)
pickle.dump(var_name, file_handle)
# always remember to close the file when you dont need it
file_handle.close()
# use 'with' keyword to make sure the file is automatically closed once you finish interacting with it
# this is equivelent with previous lines
with open(file_path + '\\' + file_name, 'wb') as fh:
    pickle.dump(var_name, fh)
# the file is automatically closed
# did I write 'var_name' twice into the file or not? why?


# now load the data
with open(file_path + '\\' + file_name, 'rb') as fh:
    data = pickle.load(fh)

print(data)
'''
as you can see, the data itself is just the content of the list. when loading, you do not get the 
variable name back. some time this is not the problem, but some times you do want to know what is 
the variable name when it is saved. how should you save the variable name as well?
probably you can use a dictionary
'''
'''
how do you save/load multiple items?
'''
