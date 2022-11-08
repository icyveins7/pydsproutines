# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 17:21:53 2022

@author: lken
"""

import sqlite3 as sq

class SqliteTemplate:
    '''
    Tables should be specified with the tablename as the key,
    and a list of lists for the columns specification.
    Each inner list should be defined as a length-of-2 list that has the 
    (column name, sqlite type) specification.
    
    Example:
      tables = {
          'tablename': [
              ["col1", "INTEGER"],
              ["col2", "REAL"]
          ]
      }
    '''
    
    tables = dict()
    
    def __init__(self, dbfilepath: str=":memory:", createTablesNow: bool=True):
        self.dbfilepath = dbfilepath # Save internally for reference
        self.con = sq.connect(dbfilepath) # Sqlite connection
        self.cur = self.con.cursor() # Cursor for statements
        if createTablesNow:
            self.createAllTables()
            
    #%% Bunch of commonly used external redirects
    def commit(self):
        self.con.commit()
        
    def close(self):
        self.con.close()
        
    #%% Table creation helpers
    def createTable(self, 
                    tablename: str,
                    columns: list,
                    ifNotExists: bool=True,
                    commitNow: bool=True):
        columns_string = ", ".join(["%s %s" % (i,j) for i, j in columns])
        stmt = "create table %s%s(%s)" % (
            "if not exists " if ifNotExists else "",
            tablename,
            columns_string
        )
        self.cur.execute(stmt)
        if commitNow:
            self.con.commit()
            
        return stmt
    
    def createAllTables(self, ifNotExists: bool=True, commitNow: bool=True):
        stmts = []
        for tablename, columns in self.tables.items():
            stmt = self.createTable(tablename, columns, ifNotExists, commitNow)
            stmts.append(stmt)
        return stmts
    
    def validateTables(self):
        stmt = "select name from sqlite_master where type='table'"
        self.cur.execute(stmt)
        current_tables = self.cur.fetchall()
        print("%s" % ", ".join([i[0] for i in current_tables]))
        
    def validateColumns(self, tablename: str):
        stmt = "select sql from sqlite_master where name=?"
        self.cur.execute(stmt, (tablename,))
        results = self.cur.fetchone()[0]
        print(results)
    
    def insertIntoTable(self, tablename: str, commitNow: bool=True, *args, **kwargs):
        # If no kwargs are supplied, we default to the ordering provided in the
        # tables variable. This is an ordered list of the variables, and should
        # be provided in the *args, in order.
        if len(kwargs) == 0:    
            stmt = "insert into %s values(%s)" % (
                tablename,
                ",".join(["?"] * len(self.tables[tablename]))
            )
            
            self.cur.execute(stmt, args)

            
        else:
            stmt = "insert into %s(%s) values(%s)" % (
                tablename,
                kwargs.keys(),
                ",".join(["?"] * len(kwargs))
            )
            
            self.cur.execute(stmt, tuple(kwargs.values()))
            
        if commitNow:
            self.con.commit()
            
        return stmt
    
    def insertIntoTableMany(self):
        pass
    
    def deleteFromTable(self):
        pass # Conditional?
        
    def deleteTable(self):
        pass
    
    def selectFromTable(self):
        pass # Conditional?
        
    
    
    
    
    
if __name__ == "__main__":
    class TestDatabase(SqliteTemplate):
        tables = {
            "table1": [
                ["col1", "INTEGER"],
                ["col2", "REAL"]
            ],
            "table2": [
                ["columnA", "BLOB"],
                ["columnB", "TEXT"]
            ]
        }
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
    db = TestDatabase()
    print(db.tables)
    db.validateTables()
    db.validateColumns("table2")
    