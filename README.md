# BlackjackRecommenderSystem
Simple Ai for playing the Blackjack game. This is still work in prgress with a "working" GUI. Entry point is in main.py

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
    <li>Computer Vision</li>
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
<p>PyTorch is an open source machine learning library based on the Torch library,[3][4][5] used for applications such as computer vision and natural language processing,[6] primarily developed by Facebook's AI Research lab (FAIR).[7][8][9] It is free and open-source software released under the Modified BSD license. Although the Python interface is more polished and the primary focus of development, PyTorch also has a C++ interface.</p>

<h2>PySimpleGUI</h2>

<p>
    PySimpleGUI wraps tkinter, Qt, WxPython and Remi so that you get all the same widgets,
     but you interact with them in a more friendly way that's common across the ports.
     <a href="https://pysimplegui.readthedocs.io/en/latest/">https://pysimplegui.readthedocs.io/en/latest/</a>
</p>
