# EPV

The concept of expected possession value ranges from simple to complex approaches.
For an overview check out Edd Webster's GitHub [1]. Some other interesting contributions and insights [2-9].

### get_EPV_grid
                (fname, fpath='grids', as_class=True, origin=None, td_object=None, team='Home')

Function to load EPV-grid from local directory. <br>

+ *fname (str)* - file name
* *fpath (str)* - directory (if not 'grids')
* *as_class (boolean)* - determines whether output is pd.DataFrame or object of EPV_grid class
* *origin (str)* - to credit the origin of EPVs
* *td_object (object of tracking data class)* - tracking data class object for plotting
* *team (str)* - team to be analyzed

**Returns** 

+ *epv (pd.DataFrame or object of EPV_grid class)* - epv data as data frame or class object

## class EPV_grid

*Attributes**



# References

[1] - https://github.com/eddwebster/football_analytics <br>
[2] - Rudd, S. (2011). A Framework for Tactical Analysis and Individual Offensive Production Assessment in Soccer Using Markov Chains. New England Symposium on Statistics in Sports. chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/http://www.nessis.org/nessis11/rudd.pdf <br>
[3] - https://youtu.be/KXSLKwADXKI <br>
[4] - Fernández De La Rosa, J., Bornn, L., & Gavaldà Mestre, R. (2022). A framework for the analytical and visual interpretation of complex spatiotemporal dynamics in soccer [Universitat Politècnica de Catalunya]. https://doi.org/10.5821/dissertation-2117-363073 <br>
[5] - Fernandez, J., & Bornn, L. (2018). Wide Open Spaces: A statistical technique for measuring space creation in professional soccer. MIT Sloan Sports Analytics Conference. <br>
[6] - Fernández, J., Bornn, L., & Cervone, D. (2021). A framework for the fine-grained evaluation of the instantaneous expected value of soccer possessions. Machine Learning, 110(6), 1389–1427. https://doi.org/10.1007/s10994-021-05989-6 <br>
[7] - Fernández, J., Bornn, L., & Cervone, D. (2019). Decomposing the Immeasurable Sport: A deep learning expected possession value framework for soccer. MIT Sloan Sports Analytics Conference, Boston. <br>
[8] - https://karun.in/blog/expected-threat.html <br>
[9] - Spearman, W. (2018). Beyond expected goals. In Proceedings of the 12th MIT sloan sports analytics conference <br>