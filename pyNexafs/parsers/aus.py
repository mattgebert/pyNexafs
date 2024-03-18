from ._base import *
import overrides
    
class AusSync(parser_base):
    
    ALLOWED_EXTENSIONS = ['.asc'] #ascii files exported from the binary .mda files generated.
    
    @classmethod
    @overrides.overrides
    def file_parser(cls, file: TextIOWrapper) -> tuple[NDArray, list[str], list[str], dict[str, Any]]:
        """Reads Australian Synchrotron .asc files.

        Parameters
        ----------
        file : TextIOWrapper
            TextIOWrapper of the datafile (i.e. open('file.asc', 'r'))

        Returns
        -------
        tuple[NDArray, list[str], dict[str, Any]]
            Returns a set of data as a numpy array, 
            labels as a list of strings,
            units as a list of strings,
            and parameters as a dictionary.

        Raises
        ------
        ValueError
            If the file is not a valid .asc file.
        """
        
        # Init vars
        data, labels, params = super().file_parser(file)
        
        # Check file type
        if not file.name.endswith('.asc'):
            raise ValueError(f"File {file.name} is not a valid .asc file.")
        
        ### Read file
        # Check header is correct
        assert file.readline() == "## mda2ascii 0.3.2 generated output\n"
        
        #skip 2 lines
        [file.readline() for i in range(2)]
        
        ## 1 Initial Parameters including MDA File v, Scan Num, Ovrl scn dim, scn size 
        for i in range(4):
            line = file.readline()[2:].strip().split(' = ', 1)
            if len(line) == 2:
                params[line[0]] = line[1]

        #skip 2 lines
        [file.readline() for i in range(2)]
        # Check PV header is correct
        assert file.readline() == "#  Extra PV: name, descr, values (, unit)\n"
        
        #skip 1 line
        file.readline()

        ## 2 Extra Parameter/Values
        line = file.readline()
        while line != '\n':
            # Discard first 11 characters and linenumber '# Extra PV 1:'.
            line = line.strip().split(":", 1)[1]
            # Separate the name, description, values and units.
            line = line.split(',')
            # Trim values and append to params in a tuple.
            params[line[0].strip()] = [line[i].strip() for i in range(1,len(line))]
            # read the newline
            line = file.readline()
            
        # skip 1 extra line
        file.readline()
                        
        ## 3 Scan Header:
        assert file.readline() == "# 1-D Scan\n"
        for i in range(3):
            line = file.readline()[2:].strip().split(' = ', 1)
            if len(line) == 2:
                params[line[0]] = line[1]
            
        #skip 1 line
        file.readline()
        
        ## 4 Scan Column properties:
        
        # Labels for columns describing data columns.
        column_types = {}
        line = file.readline()
        while line != '\n':
            line = line.strip().split(': ',1)
            col_name = line[0][3:] #ignore "#  " before name
            col_attrs = line[1].split(', ')
            column_types[col_name] = col_attrs
            # read the newline    
            line = file.readline()
        
        # skip 1 line 
        file.readline()
        
        # Column descriptions
        column_descriptions = {}
        line = file.readline()
        while line != "\n":
            # Take index info
            index = int(line[1:6].strip())
            index_line = index == 1 #Boolean to determine if on index description line.
            # Take coltype info
            desc_type = line[8:26]
            assert(desc_type[0] == "[" and desc_type[-1] == "]") #Check parenthesis of the line parameters
            desc_type = desc_type[1:-1].strip()
            # Check if valid col type
            valid = False if not index_line else True
            for coltype in column_types:
                if coltype in desc_type:
                    valid = True
                    break
            if not valid:
                raise ValueError(f"Invalid column type {desc_type} in line {line}") 

            # Take info
            desc_info = line[29:].strip().split(", ")
            column_descriptions[index] = desc_info
            labels += [desc_info[0]] if not index_line else ["Index"]
            
            # read next line
            line = file.readline()
            
        # add column data to params
        params['column_types'] = column_types
        params['column_descriptions'] = column_descriptions
            
        ## 5 Scan Values
        assert file.readline() == "# 1-D Scan Values\n"
        lines = file.readlines() #read remaining lines efficiently
        
        # Convert data to numpy array.
        data = np.loadtxt(lines, delimiter="\t") 
            # [np.loadtxt(line, delimiter="\t")
            #     for line in lines]
        data = np.array(data)
            
        return data, labels, params
    
    