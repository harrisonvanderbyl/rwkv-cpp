USE_TORCH=0 LD_LIBRARY_PATH=./lib ./RWKVCPP $(python3 -c "from tkinter import filedialog; print(filedialog.askopenfilename(initialdir='./', title='Select file', filetypes=(('rwkv files', '*.rwkv'), ('all files', '*.*'))))") $1 $2 