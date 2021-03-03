Installation
============

Setup development environment
-----------------------------

Requirements
~~~~~~~~~~~~

-  Poetry: https://python-poetry.org/docs/

After installing Poetry and cloning the project from GitHub, you should
run the following command from the root of the cloned project:

.. code:: sh

    $ poetry install

All of the project's dependencies should be installed and the project
ready for further development. **Note that Poetry creates a separate
virtual environment for your project.**

Development dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

List of NiaClass' dependencies:

+----------------+--------------+------------+
| Package        | Version      | Platform   |
+================+==============+============+
| numpy          | ^1.20.0      | All        |
+----------------+--------------+------------+
| scikit-learn   | ^0.24.1      | All        |
+----------------+--------------+------------+
| NiaPy          | ^2.0.0rc12   | All        |
+----------------+--------------+------------+
| pandas         | ^1.2.1       | All        |
+----------------+--------------+------------+

List of development dependencies:

+--------------------+-----------+------------+
| Package            | Version   | Platform   |
+====================+===========+============+
| Sphinx             | ^3.5.1    | Any        |
+--------------------+-----------+------------+
| sphinx-rtd-theme   | ^0.5.1    | Any        |
+--------------------+-----------+------------+
| coveralls          | ^3.0.1    | Any        |
+--------------------+-----------+------------+