###############################################################################
# INITIALIZATION
###############################################################################

venv: venv/touchfile

venv/touchfile: requirements.txt
	# test -d venv || virtualenv venv
	/usr/bin/python3 -m venv venv
	. venv/bin/activate; pip3 install -Ur requirements.txt
	touch venv/touchfile



hufflepuff:
	. venv/bin/activate
	python3 hufflepuff.py
