****
Installation
****

Using pip:

.. code-block:: bash
    
    pip install pyate https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz

Or to build from source:

.. code-block:: bash
    
    git clone https://github.com/kevinlu1248/pyate.git
    # or alternatively, using the gh cli
    # gh repo clone kevinlu1248/pyate
    pip install pyate

To verify the installation, run:

.. code-block:: bash

    pip show pyate

which should output something like

.. code-block:: bash

    Name: pyate
    Version: 0.4.2
    Summary: PYthon Automated Term Extraction
    Home-page: https://github.com/kevinlu1248/pyate
    Author: Kevin Lu
    Author-email: kevinlu1248@gmail.com
    License: MIT
    Location: /home/kevin/.local/lib/python3.8/site-packages
    Requires: numpy, pyahocorasick, pandas, spacy
    Required-by:

