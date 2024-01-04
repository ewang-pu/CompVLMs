I will give you an input caption describing a scene. Your task 
is to:
1. Find any verbal or spatial relationships between two 
nouns in the caption.
2. Replace the selected relationship with a new 
relationship to make a new caption.
The new caption must meet the following three 
requirements:
1. The new caption must be describing a scene that is 
as different as possible from the original scene.
2. The new caption must be fluent and grammatically 
correct.
3. The new caption must make logical sense.
Here are some examples:
Original caption: the man is in front of the building
Relationships: ["in front of"]
Selected relationship: "in front of"
New relationship: behind
New caption: the man is behind the building
Original caption: the horse is eating the grass
Relationships: ['eating']
Selected relationship: eating
New relationship: jumping over
New caption: the horse is jumping over the grass

I will attach a .json file with the original captions. For each caption, create a new modified caption. 
Return a .json file with these new captions. Remember to carefully follow the steps outlined. 

