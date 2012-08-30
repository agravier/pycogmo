grep -ilr 'import pyNN.brian' {*/*/*/,*/*/,*/,}*.py | xargs sed -i 's/import pyNN.brian/import pyNN.nest/g'
