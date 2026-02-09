Each file is named "rocks_categorization_X_Y.txt" where X is the condition number (1 = Igneous, 2 = Metamorphic, 4 = Mixed).
Each file has a header row and 400 rows corresponding to the 400 trials in the experiment.
Column 1 = condition (1 = Igneous, 2 = Metamorphic, 4 = Mixed)
Column 2 = subject ID
Column 3 = session (1 = training, 2 = test)
Column 4 = block within session
Column 5 = trial within block
Column 6 = rock category
Igneous categories: 1. Andesite, 2. Basalt, 3. Diorite, 4. Gabbro, 5. Granite, 6. Obsidian, 7. Pegmatite, 8. Peridotite, 9. Pumice, 10. Rhyolite
Metamorphic categories: 1. Amphibolite, 2. Anthracite, 3. Gneiss, 4. Hornfels, 5. Marble, 6. Migmatite, 7. Phyllite, 8. Quartzite, 9. Schist, 10. Slate
Mixed categories: 1. Basalt, 2. Diorite, 3. Obsidian, 4. Pumice, 5. Anthracite, 6. Marble, 7. Dolomite, 8. Micrite, 9. Rock Gypsum, 10. Sandstone
Column 7 = token number (1,2 = training tokens; 3,4 = test tokens)
Column 8 = subject categorization response
Column 9 = whether the response was correct (1 = correct, 0 = incorrect)
Column 10 = whether the subject received feedback (1 = feedback, 0 = no feedback)
column 11 = response time in ms