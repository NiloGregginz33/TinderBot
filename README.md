# tinderresearch
Using machine learning to try to automate my tinder these are the files that I have. This is a good way of telling girls theyre pretty if youre shy because I am.
I borrowed this chinese dataset of like 4400 faces (https://github.com/HCIILAB/SCUT-FBP5500-Database-Release). I then wrote a script that separates people into categories by using the names but I can't remember where I put it.
I then ran a simple image classifier on the pictures. I wasn't sure from testing if the algorithm was sound or not so I'm going to drop it here in case anyone wants to help it. 
I started this project when I was 19 because I thought for some reason it would help me with girls so instead of actually talking to them I did this.
The data was analysed using basic tensorflow models which I thought would train a more useful model. Theres a dataset builder if you want to augment the data or try to make your own model but the one I have posted does not incorporate any data I added.
Please I would love help with this research if you can add any suggestions, corrections, augmentations, or ideas. 
I can't seem to find the original autoswiper I made with opencv so I guess if anyone wants
to recreate that or wants to augment the dataset by creating a facial realignment function that would run on each image in preproccessing and then compiles the model or a function that
aligns the faces before inference is run would be useful I think.
Sorry some of the code can only be run on Google Colab.
Sorry some of the files reference local directories I was running this on my machine and never expected to post this online since I used to be embarrassed by my Tinder bot.
