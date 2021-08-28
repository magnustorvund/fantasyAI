    """
    The goal for the prediction model is to predict as accurately as possible the points each player will get in next (1,3, 5 or 10 i.e.) gameweeks (regression)
    Therefore, the model needs to consider the differences each position gets points for. 
    This can either be solved by a) one model (hard), or b) run 1 model specifically designed for each position. 
    Ideas to features in the prediction model:
    To mitigate the underlying random nature of football an idea is to rather use expected than actual statistics (through weights, which could be estimated by another model)
    - shots on target
    - expected goals (if striker, midfielder or defender)
    - expected clean sheets (if goalkeeper, defender or midfielder)
    - expected saves (if keeper)
    - actual saves (last match, last 7 matches, average last year etc.)
    - actual clean sheets (last match, last 7 matches etc.)
    - 
    """