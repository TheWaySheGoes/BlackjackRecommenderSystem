# BlackjackRecommenderSystem
Simple Ai for playing the Blackjack game

<h1>Initial stuff</h1>
<p>All of the dependencies are in the requirements.txt 
    You can install them with "pip install -r requirements.txt"
</p>

<h1>Structure</h1>
<p>
This project is dividied into separate modules. every module is responsible for one of the following:
</p>

<ol>
    <li>GUI</li>
    <li>Ai for card recognition</li>
    <li>Ai for next move recommendation</li>
    <li>Mouse steering </li>
    <li><
    <li>Genetic recommender system (optional)</li>
</ol>

<h1>Main modules used in the project</h1>

<h2>PyGad</h2>
<p>TorchGA is part of the PyGAD library for training PyTorch models
    using the genetic algorithm (GA). This feature is supported starting
    from PyGAD 2.10.0.
</p>
<p>
    The TorchGA project has a single module named torchga.py which has a class 
    named TorchGA for preparing an initial population of PyTorch model parameters.
</p>
<p>
    PyGAD is an open-source Python library for building the genetic algorithm and
    training machine learning algorithms. Check the library's documentation
    at Read The Docs: <a>https://pygad.readthedocs.io</a>
</p>

<h2>PyTorch</h2>
<p>...</p>

<h2>PySimpleGUI</h2>

<p>
    PySimpleGUI wraps tkinter, Qt, WxPython and Remi so that you get all the same widgets,
     but you interact with them in a more friendly way that's common across the ports.
     <a>https://pysimplegui.readthedocs.io/en/latest/</a>
</p>