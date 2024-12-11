{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Checking path: C:\\Users\\shann\\OneDrive\\Documents\\GitHub\\Computer-Vision\\Dataset\n",
      "Total Images = 87\n"
     ]
    }
   ],
   "source": [
    "from pathlib import Path\n",
    "\n",
    "IMAGE_PATH = Path(\"./Dataset\")\n",
    "print(f\"Checking path: {IMAGE_PATH.resolve()}\")  \n",
    "\n",
    "if not IMAGE_PATH.exists():\n",
    "    print(f\"The directory '{IMAGE_PATH}' does not exist.\")\n",
    "else:\n",
    "    IMAGE_PATH_LIST = list(IMAGE_PATH.glob(\"**/*.jpg\"))\n",
    "    print(f'Total Images = {len(IMAGE_PATH_LIST)}')\n",
    "\n",
    "\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
