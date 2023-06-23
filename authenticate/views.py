from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.forms import UserCreationForm, UserChangeForm, PasswordChangeForm
from django.contrib import messages 
from .forms import SignUpForm, EditProfileForm
import joblib
import numpy as np
import serial

def home(request):
	return render(request, 'authenticate/home.html', {})

def result(request):
	arduino = serial.Serial('/dev/ttyUSB0', 9600)
	print('Established serial connection to Arduino')
	arduino_data = arduino.readline()
	decoded_values = str(arduino_data[0:len(arduino_data)].decode("utf-8"))
	list_values = decoded_values.split('x')
	list_in_floats = []
	for item in list_values:
		list_in_floats.append(float(item))

	cls = joblib.load('finalized_model.sav')
	lis = []
	lis.append(request.GET['N'])
	lis.append(request.GET['P'])
	lis.append(request.GET['K'])
	lis.append(list_in_floats[0])
	lis.append(list_in_floats[1])
	lis.append(request.GET['ph'])
	lis.append(list_in_floats[2])

	ans = cls.predict([lis])
	s = ''.join(map(str, ans))
	lis = [float(x) for x in lis]
	return render(request, "authenticate/result.html", {'ans':s, 'lis':lis})

def login_user(request):
	if request.method == 'POST':
		username = request.POST['username']
		password = request.POST['password']
		user = authenticate(request, username=username, password=password)
		if user is not None:
			login(request, user)
			messages.success(request, ('You Have Been Logged In!'))
			return redirect('home')

		else:
			messages.success(request, ('Error Logging In - Please Try Again...'))
			return redirect('login')
	else:
		return render(request, 'authenticate/login.html', {})

def logout_user(request):
	logout(request)
	messages.success(request, ('You Have Been Logged Out...'))
	return redirect('home')

def register_user(request):
	if request.method == 'POST':
		form = SignUpForm(request.POST)
		if form.is_valid():
			form.save()
			username = form.cleaned_data['username']
			password = form.cleaned_data['password1']
			user = authenticate(username=username, password=password)
			login(request, user)
			messages.success(request, ('You Have Registered...'))
			return redirect('home')
	else:
		form = SignUpForm()
	
	context = {'form': form}
	return render(request, 'authenticate/register.html', context)



def edit_profile(request):
	if request.method == 'POST':
		form = EditProfileForm(request.POST, instance=request.user)
		if form.is_valid():
			form.save()
			messages.success(request, ('You Have Edited Your Profile...'))
			return redirect('home')
	else:
		form = EditProfileForm(instance=request.user)
	context = {'form': form}
	return render(request, 'authenticate/edit_profile.html', context)

def change_password(request):
	if request.method == 'POST':
		form = PasswordChangeForm(data=request.POST, user=request.user)
		if form.is_valid():
			form.save()
			update_session_auth_hash(request, form.user)
			messages.success(request, ('You Have Edited Your Password...'))
			return redirect('home')
	else:
		form = PasswordChangeForm(user=request.user)
	
	context = {'form': form}
	return render(request, 'authenticate/change_password.html', context)

