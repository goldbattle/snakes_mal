



### Install Instructions


1. Create your virtual enviroment
    - `pip install virtualenv`
    - `virtualenv venv`
    - `source venv/bin/activate`

2. Install dependencies
    - `pip3 install pyqtgraph` (for fast plotting)
    - `pip3 install pyqt5` (for fast plotting)
    - `pip3 install numpy`
    - `pip3 install gym` (open ai gym)
    - Ensure you have cuda 9: https://www.tensorflow.org/install/gpu
    - `pip3 install tensorflow-gpu==1.9.0` (cuda version 9.0)


3. Install mgym
    - A local copy has been included in this repo 
    - https://github.com/cjm715/mgym
    - `cd mgym`
    - `pip install -e .`
    - `cd ..`
    - It should be installed

4. Install maddpg
    - A local copy has been included in this repo
    - https://github.com/openai/maddpg
    - `cd maddpg`
    - `pip install -e .`
    - `cd ..`
    - It should be installed



5. Test our installs:
    - I have included a few test scripts that have minimal example.
    - Run each test script and see if you get any loading errors
    - `python3 test_mgym.py`
    - `python3 test_maddpg.py`
    - `python3 test_tensorflow.py`


