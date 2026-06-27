import json
import os

path = r"f:\Learning\DL\Learning\Deep-Learning\functional_model\end-to-end-example.ipynb"

print(f"Reading {path}...")
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Code to insert
wrapper_code = [
    "from tensorflow.keras.utils import Sequence\n",
    "\n",
    "class DictTargetWrapper(Sequence):\n",
    "    def __init__(self, generator):\n",
    "        self.generator = generator\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.generator)\n",
    "\n",
    "    def __getitem__(self, index):\n",
    "        X, y = self.generator[index]\n",
    "        # y is [age_batch, gender_batch]\n",
    "        return X, {'age': y[0], 'gender': y[1]}\n",
    "\n",
    "train_gen_wrapped = DictTargetWrapper(train_generator)\n",
    "test_gen_wrapped = DictTargetWrapper(test_generator)\n"
]

# Find model.fit cell
fit_cell_index = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "model.fit" in source:
            fit_cell_index = i
            break

if fit_cell_index != -1:
    print(f"Found model.fit at cell {fit_cell_index}")
    
    # Check if we already inserted the wrapper (simple check)
    prev_cell = nb['cells'][fit_cell_index - 1]
    if prev_cell['cell_type'] == 'code' and "class DictTargetWrapper" in "".join(prev_cell['source']):
        print("Wrapper already inserted. Skipping insertion.")
    else:
        # Create new cell
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "id": "wrapper_fix_cell",
            "metadata": {},
            "outputs": [],
            "source": wrapper_code
        }
        
        # Insert before fit cell
        nb['cells'].insert(fit_cell_index, new_cell)
        print("Inserted wrapper cell.")
        
        # Update fit cell index since we shifted list
        fit_cell_index += 1

    # Update fit cell
    fit_cell = nb['cells'][fit_cell_index]
    new_source = []
    changed = False
    for line in fit_cell['source']:
        original_line = line
        if "model.fit" in line:
            # Only replace if not already wrapped
            if "train_gen_wrapped" not in line:
                line = line.replace("train_generator", "train_gen_wrapped")
                changed = True
            if "validation_data=test_generator" in line:
                 line = line.replace("validation_data=test_generator", "validation_data=test_gen_wrapped")
                 changed = True
        new_source.append(line)
    
    if changed:
        fit_cell['source'] = new_source
        print("Updated model.fit arguments.")
    else:
        print("model.fit arguments already updated or no change needed.")
    
    print("Saving changes...")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Done.")

else:
    print("Could not find model.fit cell.")
