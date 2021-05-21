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

<p>PySimpleGUI wraps tkinter, Qt, WxPython and Remi so that you get all the same widgets, but you interact with them in a more friendly way that's common across the ports.

    What does a wrapper do (Yo! PSG in the house!)? It does the layout, boilerplate code, creates and manages the GUI Widgets for you and presents you with a simple, efficient interface. Most importantly, it maps the Widgets in tkinter/Qt/Wx/Remi into PySimpleGUI Elements. Finally, it replaces the GUIs' event loop with one of our own.
    
    You've seen examples of the code already. The big deal of all this is that anyone can create a GUI simply and quickly that matches GUIs written in the native GUI framework. You can create complex layouts with complex element interactions. And, that code you wrote to run on tkinter will also run on Qt by changing your import statement.
    
    If you want a deeper explanation about the architecture of PySimpleGUI, you'll find it on ReadTheDocs in the same document as the Readme & Cookbook. There is a tab at the top with labels for each document. <a>https://pysimplegui.readthedocs.io/en/latest/</a></p>