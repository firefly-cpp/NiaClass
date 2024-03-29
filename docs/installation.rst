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
| niapy          | ^2.0.4       | All        |
+----------------+--------------+------------+
| pandas         | ^1.5.0       | All        |
+----------------+--------------+------------+
| numpy          | ^1.26.0      | All        |
+----------------+--------------+------------+
| scikit-learn   | ^1.2.0       | All        |
+----------------+--------------+------------+

List of development dependencies:

+--------------------+-----------+------------+
| Package            | Version   | Platform   |
+====================+===========+============+
| coveralls          | ^3.0.1    | Any        |
+--------------------+-----------+------------+
| Sphinx             | ^3.5.1    | Any        |
+--------------------+-----------+------------+
| sphinx-rtd-theme   | ^0.5.1    | Any        |
+--------------------+-----------+------------+
| autoflake          | ^1.4      | Any        |
+--------------------+-----------+------------+
| black              | ^21.5b1   | Any        |
+--------------------+-----------+------------+
| pre-commit         | ^2.13.0   | Any        |
+--------------------+-----------+------------+
| pytest             | ^6.2      | Any        |
+--------------------+-----------+------------+