#include "mcts.h"
#include "testsimulator.h"
#include <math.h>

#include <algorithm>

using namespace std;
using namespace UTILS;

//-----------------------------------------------------------------------------

MCTS::PARAMS::PARAMS()
:   Verbose(0),
    MaxDepth(100),
    NumSimulations(1000),
    NumStartStates(1000),
    UseTransforms(true),
    NumTransforms(0),
    MaxAttempts(0),
    ExpandCount(1),
    ExplorationConstant(1),
    UseRave(false),
    RaveDiscount(1.0),
    RaveConstant(0.01),
    DisableTree(false),
	HGreedy(1)
{
}

MCTS::MCTS(const SIMULATOR& simulator, const PARAMS& params)
:   Simulator(simulator),
    Params(params),
    TreeDepth(0)
{
    VNODE::NumChildren = Simulator.GetNumActions();
    QNODE::NumChildren = Simulator.GetNumObservations();

    Root = ExpandNode(Simulator.CreateStartState());

    for (int i = 0; i < Params.NumStartStates; i++)
        Root->Beliefs().AddSample(Simulator.CreateStartState());
}

MCTS::~MCTS()
{
    VNODE::Free(Root, Simulator);
    VNODE::FreeAll();
}

bool MCTS::Update(int action, int observation, double reward)
{
    History.Add(action, observation);
    BELIEF_STATE beliefs;

    // Find matching vnode from the rest of the tree
    QNODE& qnode = Root->Child(action);
    VNODE* vnode = qnode.Child(observation);
    if (vnode)
    {
        if (Params.Verbose >= 1)
            cout << "Matched " << vnode->Beliefs().GetNumSamples() << " states" << endl;
        beliefs.Copy(vnode->Beliefs(), Simulator);
    }
    else
    {
        if (Params.Verbose >= 1)
            cout << "No matching node found" << endl;
    }

    // Generate transformed states to avoid particle deprivation
    if (Params.UseTransforms)
        AddTransforms(Root, beliefs);

    // If we still have no particles, fail
    if (beliefs.Empty() && (!vnode || vnode->Beliefs().Empty()))
        return false;

    if (Params.Verbose >= 1)
        Simulator.DisplayBeliefs(beliefs, cout);

    // Find a state to initialise prior (only requires fully observed state)
    const STATE* state = 0;
    if (vnode && !vnode->Beliefs().Empty())
        state = vnode->Beliefs().GetSample(0);
    else
        state = beliefs.GetSample(0);

    // Delete old tree and create new root
    VNODE::Free(Root, Simulator);
    VNODE* newRoot = ExpandNode(state);
    newRoot->Beliefs() = beliefs;
    Root = newRoot;
    return true;
}

int MCTS::SelectAction()
{
    if (Params.DisableTree)
        RolloutSearch();
    else
        UCTSearch();
    return GreedyUCB(Root, false);
}

void MCTS::RolloutSearch()
{
	std::vector<double> totals(Simulator.GetNumActions(), 0.0);
	int historyDepth = History.Size();
	std::vector<int> legal;
	assert(BeliefState().GetNumSamples() > 0);
	Simulator.GenerateLegal(*BeliefState().GetSample(0), GetHistory(), legal, GetStatus());
	random_shuffle(legal.begin(), legal.end());

	for (int i = 0; i < Params.NumSimulations; i++)
	{
		int action = legal[i % legal.size()];
		STATE* state = Root->Beliefs().CreateSample(Simulator);
		Simulator.Validate(*state);

		int observation;
		double immediateReward, delayedReward, totalReward;
		bool terminal = Simulator.Step(*state, action, observation, immediateReward);

		VNODE*& vnode = Root->Child(action).Child(observation);
		if (!vnode && !terminal)
		{
			vnode = ExpandNode(state);
			AddSample(vnode, *state);
		}
		History.Add(action, observation);

		delayedReward = Rollout(*state);
		totalReward = immediateReward + Simulator.GetDiscount() * delayedReward;
		Root->Child(action).Value.Add(totalReward);

		Simulator.FreeState(state);
		History.Truncate(historyDepth);
	}
}

void MCTS::UCTSearch()
{
    ClearStatistics();
    int historyDepth = History.Size();

    for (int n = 0; n < Params.NumSimulations; n++)
    {
        STATE* state = Root->Beliefs().CreateSample(Simulator);
        Simulator.Validate(*state);
        Status.Phase = SIMULATOR::STATUS::TREE;
        if (Params.Verbose >= 2)
        {
            cout << "Starting simulation" << endl;
            Simulator.DisplayState(*state, cout);
        }

        TreeDepth = 0;
        PeakTreeDepth = 0;

        vector <double> totalRewardVector = SimulateV(*state, Root);
        for (int i = 0; i < totalRewardVector.size(); i++)
        {
        	StatTotalReward.Add(totalRewardVector[i]);
        	StatTreeDepth.Add(PeakTreeDepth);
        }

//        if (Params.Verbose >= 2)
//            cout << "Total reward = " << totalReward << endl;
        if (Params.Verbose >= 3)
            DisplayValue(4, cout);

        Simulator.FreeState(state);
        History.Truncate(historyDepth);
    }

    DisplayStatistics(cout);
}

vector <double> MCTS::ExpandHDepth(STATE& state, VNODE* vnode, int depth)
{
    int observation;
    double immediateReward, delayedReward = 0, totalReward = 0;
    vector <double> totalRewardVector;

    totalRewardVector.clear();

    int historyDepth = History.Size();

//	STATE* state = Root->Beliefs().CreateSample(Simulator);

	depth--;

    for (int action = 0; action < Simulator.GetNumActions(); action++)
    {
    	//because it starts from the same vnode, it should share the same state
        STATE* newstate = Simulator.Copy(state);

        delayedReward = 0;

    	QNODE& qnode = vnode->Child(action);

    	if (Simulator.HasAlpha())
    		Simulator.UpdateAlpha(qnode, *newstate);

    	bool terminal = Simulator.Step(*newstate, action, observation, immediateReward);
    	assert(observation >= 0 && observation < Simulator.GetNumObservations());
    	History.Add(action, observation);

    	History.Truncate(historyDepth);

//    	cout << "TreeDepth; " << TreeDepth << "\n";
//        if (TreeDepth >= 1)
//        	AddSample(vnode, state);

        if (terminal == true)
        {
    		totalReward = immediateReward + Simulator.GetDiscount() * delayedReward;
    		qnode.Value.Add(totalReward);
    		//
    		//and vnode also
    		vnode->Value.Add(totalReward);
    		AddRave(vnode, totalReward);

    		totalRewardVector.push_back(totalReward);
        }
        //in case it does not have enough expand count
        else
        {
        	while (qnode.Value.GetCount() < Params.ExpandCount)
        	{
        		STATE* newRolloutState = Simulator.Copy(*newstate);
        		TreeDepth++;

        		//after rollout we need to update qnode
        		delayedReward = Rollout(*newRolloutState);

        		TreeDepth--;

        		totalReward = immediateReward + Simulator.GetDiscount() * delayedReward;
        		qnode.Value.Add(totalReward);
        		//
        		//and vnode also
        		vnode->Value.Add(totalReward);
        		AddRave(vnode, totalReward);

        		totalRewardVector.push_back(totalReward);


        		Simulator.FreeState(newRolloutState);
        	}

        	//we only expand if depth >= 1
        	if (depth >= 1)
        	{
        		VNODE*& newvnode = qnode.Child(observation);

        		if (!newvnode)
        			newvnode = ExpandNode(newstate);


        		//if continue rollout, we need to update vnode
        		vector <double> tmpList;

        		tmpList.clear();

        		TreeDepth++;

        		tmpList = ExpandHDepth(*newstate, newvnode, depth);

        		TreeDepth--;


        		for (int i = 0; i < tmpList.size(); i++)
        		{
        			vnode->Value.Add(tmpList[i]);
        			AddRave(vnode, tmpList[i]);

        			totalRewardVector.push_back(tmpList[i]);
        		}
        	}
        }

    	//free it
    	Simulator.FreeState(newstate);
    	History.Truncate(historyDepth);

    }

//    double averageReward = vnode->Value.GetValue()/vnode->Value.GetCount();
	return totalRewardVector;
}

//
//
//GreedyQNode MCTS::ExpandHGreedyDepth(STATE& state, VNODE* vnode, int depth)
//{
//    int observation;
//    double immediateReward, delayedReward = 0, totalReward = 0;
//
//    int historyDepth = History.Size();
//
//    GreedyQNode greedyQ, tmpQ, maxGreedyQ;
//
//    maxGreedyQ.action = 0;
//    maxGreedyQ.qValue = 0;
//    maxGreedyQ.greedyValue = 0;
//
//
////	STATE* state = Root->Beliefs().CreateSample(Simulator);
//
//    depth--;
//
//    for (int action = 0; action < Simulator.GetNumActions(); action++)
//    {
//    	//because it starts from the same vnode, it should share the same state
//        STATE* newstate = Simulator.Copy(state);
//
//        delayedReward = 0;
//
//    	QNODE& qnode = vnode->Child(action);
//
//    	if (Simulator.HasAlpha())
//    		Simulator.UpdateAlpha(qnode, *newstate);
//
//    	bool terminal = Simulator.Step(*newstate, action, observation, immediateReward);
//    	assert(observation >= 0 && observation < Simulator.GetNumObservations());
//    	History.Add(action, observation);
//
//    	History.Truncate(historyDepth);
//
////        if (TreeDepth == 1)
////        	AddSample(vnode, *newstate);
//
//        if (terminal == true)
//        {
//    		totalReward = immediateReward + Simulator.GetDiscount() * delayedReward;
//    		qnode.Value.Add(totalReward);
//    		//
//    		//and vnode also
//    		vnode->Value.Add(totalReward);
//    		AddRave(vnode, totalReward);
//        }
//        //in case it does not have enough expand count
//        else
//        {
//        	while (qnode.Value.GetCount() < Params.ExpandCount)
//        	{
//        		STATE* newRolloutState = Simulator.Copy(*newstate);
//        		TreeDepth++;
//
//        		//after rollout we need to update qnode
//        		delayedReward = Rollout(*newRolloutState);
//
//        		TreeDepth--;
//
//        		totalReward = immediateReward + Simulator.GetDiscount() * delayedReward;
//        		qnode.Value.Add(totalReward);
//        		//
//        		//and vnode also
//        		vnode->Value.Add(totalReward);
//        		AddRave(vnode, totalReward);
//
//        		Simulator.FreeState(newRolloutState);
//        	}
//
//        	//we only expand if depth >= 1
//        	if (depth >= 1)
//        	{
//        		VNODE*& newvnode = qnode.Child(observation);
//
//        		if (!newvnode)
//        			newvnode = ExpandNode(newstate);
//
//
//        		//if continue rollout, we need to update vnode
//        		tmpQ = ExpandHGreedyDepth(*newstate, newvnode, depth);
//
//
//        		vnode->Value.Add(tmpQ.qValue);
//        		AddRave(vnode, tmpQ.qValue);
//        	}
//        }
//
//    	//free it
//    	Simulator.FreeState(newstate);
//    	History.Truncate(historyDepth);
//    }
//
//    //    if (depth == 0)
//    //    {
//    //    	greedyQ.qValue = vnode->Value.GetValue();
//    //    	GreedyQNode tmp = GreedyHUCB(vnode, true);
//    //    	greedyQ.action = tmp.action;
//    //    	greedyQ.greedyValue = tmp.greedyValue;
//    //    	return greedyQ;
//    //    }
//    //    else
//    //    {
//    //    	greedyQ.qValue = vnode->Value.GetValue();
//    //    }
////    return vnode->Value.GetValue();
//
//    greedyQ = GreedyHUCB(vnode, true);
//
//
//    return greedyQ;
//}
//
//
//


GreedyQNode MCTS::ExpandHGreedyDepth(STATE& state, VNODE* vnode, int depth)
{
    int observation;
    double immediateReward, delayedReward = 0, totalReward = 0;

    int historyDepth = History.Size();

    GreedyQNode tmpQ, maxGreedyQ;

    maxGreedyQ.action = 0;
    maxGreedyQ.qValue = 0;
    maxGreedyQ.greedyValue = 0;


//	STATE* state = Root->Beliefs().CreateSample(Simulator);

    depth--;

    for (int action = 0; action < Simulator.GetNumActions(); action++)
    {
    	//because it starts from the same vnode, it should share the same state
        STATE* newstate = Simulator.Copy(state);

        delayedReward = 0;

    	QNODE& qnode = vnode->Child(action);

    	if (Simulator.HasAlpha())
    		Simulator.UpdateAlpha(qnode, *newstate);

    	bool terminal = Simulator.Step(*newstate, action, observation, immediateReward);
    	assert(observation >= 0 && observation < Simulator.GetNumObservations());
    	History.Add(action, observation);

    	History.Truncate(historyDepth);

//        if (TreeDepth == 1)
//        	AddSample(vnode, state);

        if (terminal == true)
        {
    		totalReward = immediateReward + Simulator.GetDiscount() * delayedReward;
    		qnode.Value.Add(totalReward);
    		//
    		//and vnode also
    		vnode->Value.Add(totalReward);
    		AddRave(vnode, totalReward);
        }
        //in case it does not have enough expand count
        else
        {
        	while (qnode.Value.GetCount() < Params.ExpandCount)
        	{
        		STATE* newRolloutState = Simulator.Copy(*newstate);

        		//after rollout we need to update qnode
        		delayedReward = Rollout(*newRolloutState);


        		totalReward = immediateReward + Simulator.GetDiscount() * delayedReward;
        		qnode.Value.Add(totalReward);
        		//
        		//and vnode also
        		vnode->Value.Add(totalReward);
        		AddRave(vnode, totalReward);

        		Simulator.FreeState(newRolloutState);
        	}

        	//we only expand if depth >= 1
        	if (depth >= 1)
        	{
        		VNODE*& newvnode = qnode.Child(observation);

        		if (!newvnode)
        			newvnode = ExpandNode(newstate);


        		//if continue rollout, we need to update vnode
        		tmpQ = ExpandHGreedyDepth(*newstate, newvnode, depth);

        		vnode->Value.Add(tmpQ.qValue);
        		AddRave(vnode, tmpQ.qValue);

//        		if (depth == (Params.HGreedy - 1) && (maxGreedyQ.greedyValue < tmpQ.greedyValue))
//        		{
//        			maxGreedyQ.action = action;
//        			maxGreedyQ.greedyValue = tmpQ.greedyValue;
//        		}
        		if (maxGreedyQ.greedyValue < tmpQ.greedyValue)
        		{
        			maxGreedyQ.action = action;
        			maxGreedyQ.greedyValue = tmpQ.greedyValue;
        		}
        	}
        }

    	//free it
    	Simulator.FreeState(newstate);
    	History.Truncate(historyDepth);
    }

    if (depth == 0)
    {
    	maxGreedyQ.qValue = vnode->Value.GetValue();
    	GreedyQNode tmp = GreedyHUCB(vnode, true);
    	maxGreedyQ.action = tmp.action;
    	maxGreedyQ.greedyValue = tmp.greedyValue;
    }
    else
    {
    	maxGreedyQ.qValue = vnode->Value.GetValue();
    }

//    greedyQ = GreedyHUCB(vnode, true);


    return maxGreedyQ;
}




vector <double> MCTS::SimulateV(STATE& state, VNODE* vnode)
{
	int h = Params.HGreedy;

	//make copy of the current state
    STATE* newstate = Simulator.Copy(state);

    vector <double> totalRewardVector;

    totalRewardVector.clear();

    vector <double> tmpVector;

    totalRewardVector = ExpandHDepth(*newstate, vnode, h);

//	for (int i = 0; i < totalRewardVector.size(); i++)
//	{
//		vnode->Value.Add(totalRewardVector[i]);
//		AddRave(vnode, totalRewardVector[i]);
//	}

    //expand the current vnode
//    GreedyQNode greedyQ = ExpandHGreedyDepth(*newstate, vnode, h);

    //free the state
    Simulator.FreeState(newstate);

	int action = GreedyUCB(vnode, true);

//    int action = greedyQ.action;

    PeakTreeDepth = TreeDepth;

    if (TreeDepth >= Params.MaxDepth) // search horizon reached
        return totalRewardVector;

    if (TreeDepth == 1)
        AddSample(vnode, state);

    QNODE& qnode = vnode->Child(action);

    tmpVector.clear();
    tmpVector = SimulateQ(state, qnode, action);


	for (int i = 0; i < tmpVector.size(); i++)
	{
		vnode->Value.Add(tmpVector[i]);
		AddRave(vnode, tmpVector[i]);
	    totalRewardVector.push_back(tmpVector[i]);
	}

    return totalRewardVector;
}

vector <double> MCTS::SimulateQ(STATE& state, QNODE& qnode, int action)
{
    int observation;
    double immediateReward, delayedReward = 0;

    vector <double> totalRewardVector;

    totalRewardVector.clear();

    vector <double> tmpVector;

    tmpVector.clear();

    tmpVector.push_back(delayedReward);

    if (Simulator.HasAlpha())
        Simulator.UpdateAlpha(qnode, state);

    bool terminal = Simulator.Step(state, action, observation, immediateReward);

    assert(observation >= 0 && observation < Simulator.GetNumObservations());
    History.Add(action, observation);

    if (Params.Verbose >= 3)
    {
        Simulator.DisplayAction(action, cout);
        Simulator.DisplayObservation(state, observation, cout);
        Simulator.DisplayReward(immediateReward, cout);
        Simulator.DisplayState(state, cout);
        cout << "terminal: " << terminal << "\n";
    }

    VNODE*& vnode = qnode.Child(observation);
    if (!vnode && !terminal && qnode.Value.GetCount() >= Params.ExpandCount)
    {
        vnode = ExpandNode(&state);
    }

    if (!terminal)
    {
        TreeDepth++;

        if (vnode)
        {
        	tmpVector = SimulateV(state, vnode);
        }
        else
        {
            delayedReward = Rollout(state);
            tmpVector.push_back(delayedReward);
        }
        TreeDepth--;
    }

	for (int i = 0; i < tmpVector.size(); i++)
	{
		double totalReward = immediateReward + Simulator.GetDiscount() * tmpVector[i];
		qnode.Value.Add(totalReward);

		totalRewardVector.push_back(totalReward);
	}
    return totalRewardVector;
}

void MCTS::AddRave(VNODE* vnode, double totalReward)
{
	if (Params.UseRave == false)
		return;
    double totalDiscount = 1.0;
    for (int t = TreeDepth; t < History.Size(); ++t)
    {
        QNODE& qnode = vnode->Child(History[t].Action);
        qnode.AMAF.Add(totalReward, totalDiscount);
        totalDiscount *= Params.RaveDiscount;
    }
}

VNODE* MCTS::ExpandNode(const STATE* state)
{
    VNODE* vnode = VNODE::Create();
    vnode->Value.Set(0, 0);
    Simulator.Prior(state, History, vnode, Status);

    if (Params.Verbose >= 2)
    {
        cout << "Expanding node: ";
        History.Display(cout);
        cout << endl;
    }

    return vnode;
}

void MCTS::AddSample(VNODE* node, const STATE& state)
{
    STATE* sample = Simulator.Copy(state);
    node->Beliefs().AddSample(sample);
    if (Params.Verbose >= 2)
    {
        cout << "Adding sample:" << endl;
        Simulator.DisplayState(*sample, cout);
    }
}

GreedyQNode MCTS::GreedyHUCB(VNODE* vnode, bool ucb) const
{
    static vector<int> besta;
    besta.clear();
    double bestq = -Infinity;
    int N = vnode->Value.GetCount();
    double logN = log(N + 1);
    bool hasalpha = Simulator.HasAlpha();

    GreedyQNode hgreedyUCB;
    hgreedyUCB.qValue = vnode->Value.GetValue();
    hgreedyUCB.action = 0;
    hgreedyUCB.greedyValue = -2*Infinity;

    for (int action = 0; action < Simulator.GetNumActions(); action++)
    {
    	double q, alphaq;
    	int n, alphan;

    	QNODE& qnode = vnode->Child(action);
    	q = qnode.Value.GetValue();
    	n = qnode.Value.GetCount();

    	if (Params.UseRave && qnode.AMAF.GetCount() > 0)
    	{
    		double n2 = qnode.AMAF.GetCount();
    		double beta = n2 / (n + n2 + Params.RaveConstant * n * n2);
    		q = (1.0 - beta) * q + beta * qnode.AMAF.GetValue();
    	}

    	if (hasalpha && n > 0)
    	{
    		Simulator.AlphaValue(qnode, alphaq, alphan);
    		q = (n * q + alphan * alphaq) / (n + alphan);
    		//cout << "N = " << n << ", alphaN = " << alphan << endl;
    		//cout << "Q = " << q << ", alphaQ = " << alphaq << endl;
    	}

    	if (ucb)
    		q += FastUCB(N, n, logN);

    	if (q >= bestq)
    	{
    		if (q > bestq)
    			besta.clear();
    		bestq = q;
    		besta.push_back(action);

    		hgreedyUCB.greedyValue = bestq;
    	}
    }

    assert(!besta.empty());
	hgreedyUCB.action = besta[Random(besta.size())];
    return hgreedyUCB;
}



int MCTS::GreedyUCB(VNODE* vnode, bool ucb) const
{
    static vector<int> besta;
    besta.clear();
    double bestq = -Infinity;
    int N = vnode->Value.GetCount();
    double logN = log(N + 1);
    bool hasalpha = Simulator.HasAlpha();

    for (int action = 0; action < Simulator.GetNumActions(); action++)
    {
    	double q, alphaq;
    	int n, alphan;

    	QNODE& qnode = vnode->Child(action);
    	q = qnode.Value.GetValue();
    	n = qnode.Value.GetCount();

    	if (Params.UseRave && qnode.AMAF.GetCount() > 0)
    	{
    		double n2 = qnode.AMAF.GetCount();
    		double beta = n2 / (n + n2 + Params.RaveConstant * n * n2);
    		q = (1.0 - beta) * q + beta * qnode.AMAF.GetValue();
    	}

    	if (hasalpha && n > 0)
    	{
    		Simulator.AlphaValue(qnode, alphaq, alphan);
    		q = (n * q + alphan * alphaq) / (n + alphan);
    		//cout << "N = " << n << ", alphaN = " << alphan << endl;
    		//cout << "Q = " << q << ", alphaQ = " << alphaq << endl;
    	}

    	if (ucb)
    		q += FastUCB(N, n, logN);

    	if (q >= bestq)
    	{
    		if (q > bestq)
    			besta.clear();
    		bestq = q;
    		besta.push_back(action);
    	}
    }

    assert(!besta.empty());
    return besta[Random(besta.size())];
}



double MCTS::Rollout(STATE& state)
{
    Status.Phase = SIMULATOR::STATUS::ROLLOUT;
    if (Params.Verbose >= 3)
        cout << "Starting rollout" << endl;

    double totalReward = 0.0;
    double discount = 1.0;
    bool terminal = false;
    int numSteps;
    for (numSteps = 0; numSteps + TreeDepth < Params.MaxDepth && !terminal; ++numSteps)
    {
        int observation;
        double reward;

        int action = Simulator.SelectRandom(state, History, Status);
        terminal = Simulator.Step(state, action, observation, reward);
        History.Add(action, observation);

        if (Params.Verbose >= 4)
        {
            Simulator.DisplayAction(action, cout);
            Simulator.DisplayObservation(state, observation, cout);
            Simulator.DisplayReward(reward, cout);
            Simulator.DisplayState(state, cout);
        }

        totalReward += reward * discount;
        discount *= Simulator.GetDiscount();
    }

    StatRolloutDepth.Add(numSteps);
    if (Params.Verbose >= 3)
        cout << "Ending rollout after " << numSteps
            << " steps, with total reward " << totalReward << endl;
    return totalReward;
}

double MCTS::RolloutHGreedy(STATE& state)
{
    Status.Phase = SIMULATOR::STATUS::ROLLOUT;
    if (Params.Verbose >= 3)
        cout << "Starting rollout" << endl;

    double totalReward = 0.0;
    double discount = 1.0;
    bool terminal = false;
    int numSteps;
    for (numSteps = 0; numSteps < Params.HGreedy && !terminal; ++numSteps)
    {
        int observation;
        double reward;

        int action = Simulator.SelectRandom(state, History, Status);
        terminal = Simulator.Step(state, action, observation, reward);
        History.Add(action, observation);

        if (Params.Verbose >= 4)
        {
            Simulator.DisplayAction(action, cout);
            Simulator.DisplayObservation(state, observation, cout);
            Simulator.DisplayReward(reward, cout);
            Simulator.DisplayState(state, cout);
        }

        totalReward += reward * discount;
        discount *= Simulator.GetDiscount();
    }

    StatRolloutDepth.Add(numSteps);
    if (Params.Verbose >= 3)
        cout << "Ending rollout after " << numSteps
            << " steps, with total reward " << totalReward << endl;
    return totalReward;
}


void MCTS::AddTransforms(VNODE* root, BELIEF_STATE& beliefs)
{
    int attempts = 0, added = 0;

    // Local transformations of state that are consistent with history
    while (added < Params.NumTransforms && attempts < Params.MaxAttempts)
    {
        STATE* transform = CreateTransform();
        if (transform)
        {
            beliefs.AddSample(transform);
            added++;
        }
        attempts++;
    }

    if (Params.Verbose >= 1)
    {
        cout << "Created " << added << " local transformations out of "
            << attempts << " attempts" << endl;
    }
}

STATE* MCTS::CreateTransform() const
{
    int stepObs;
    double stepReward;

    STATE* state = Root->Beliefs().CreateSample(Simulator);
    Simulator.Step(*state, History.Back().Action, stepObs, stepReward);
    if (Simulator.LocalMove(*state, History, stepObs, Status))
        return state;
    Simulator.FreeState(state);
    return 0;
}

double MCTS::UCB[UCB_N][UCB_n];
bool MCTS::InitialisedFastUCB = true;

void MCTS::InitFastUCB(double exploration)
{
    cout << "Initialising fast UCB table... ";
    for (int N = 0; N < UCB_N; ++N)
        for (int n = 0; n < UCB_n; ++n)
            if (n == 0)
                UCB[N][n] = Infinity;
            else
                UCB[N][n] = exploration * sqrt(log(N + 1) / n);
    cout << "done" << endl;
    InitialisedFastUCB = true;
}

inline double MCTS::FastUCB(int N, int n, double logN) const
{
    if (InitialisedFastUCB && N < UCB_N && n < UCB_n)
        return UCB[N][n];

    if (n == 0)
        return Infinity;
    else
        return Params.ExplorationConstant * sqrt(logN / n);
}

void MCTS::ClearStatistics()
{
    StatTreeDepth.Clear();
    StatRolloutDepth.Clear();
    StatTotalReward.Clear();
}

void MCTS::DisplayStatistics(ostream& ostr) const
{
    if (Params.Verbose >= 1)
    {
        StatTreeDepth.Print("Tree depth", ostr);
        StatRolloutDepth.Print("Rollout depth", ostr);
        StatTotalReward.Print("Total reward", ostr);
    }

    if (Params.Verbose >= 2)
    {
        ostr << "Policy after " << Params.NumSimulations << " simulations" << endl;
        DisplayPolicy(6, ostr);
        ostr << "Values after " << Params.NumSimulations << " simulations" << endl;
        DisplayValue(6, ostr);
    }
}

void MCTS::DisplayValue(int depth, ostream& ostr) const
{
    HISTORY history;
    ostr << "MCTS Values:" << endl;
    Root->DisplayValue(history, depth, ostr);
}

void MCTS::DisplayPolicy(int depth, ostream& ostr) const
{
    HISTORY history;
    ostr << "MCTS Policy:" << endl;
    Root->DisplayPolicy(history, depth, ostr);
}

//-----------------------------------------------------------------------------

void MCTS::UnitTest()
{
    UnitTestGreedy();
    UnitTestUCB();
    UnitTestRollout();
    for (int depth = 1; depth <= 3; ++depth)
        UnitTestSearch(depth);
}

void MCTS::UnitTestGreedy()
{
    TEST_SIMULATOR testSimulator(5, 5, 0);
    PARAMS params;
    MCTS mcts(testSimulator, params);
    int numAct = testSimulator.GetNumActions();
    int numObs = testSimulator.GetNumObservations();

    VNODE* vnode = mcts.ExpandNode(testSimulator.CreateStartState());
    vnode->Value.Set(1, 0);
    vnode->Child(0).Value.Set(0, 1);
    for (int action = 1; action < numAct; action++)
        vnode->Child(action).Value.Set(0, 0);
    assert(mcts.GreedyUCB(vnode, false) == 0);
}

void MCTS::UnitTestUCB()
{
    TEST_SIMULATOR testSimulator(5, 5, 0);
    PARAMS params;
    MCTS mcts(testSimulator, params);
    int numAct = testSimulator.GetNumActions();
    int numObs = testSimulator.GetNumObservations();

    // With equal value, action with lowest count is selected
    VNODE* vnode1 = mcts.ExpandNode(testSimulator.CreateStartState());
    vnode1->Value.Set(1, 0);
    for (int action = 0; action < numAct; action++)
        if (action == 3)
            vnode1->Child(action).Value.Set(99, 0);
        else
            vnode1->Child(action).Value.Set(100 + action, 0);
    assert(mcts.GreedyUCB(vnode1, true) == 3);

    // With high counts, action with highest value is selected
    VNODE* vnode2 = mcts.ExpandNode(testSimulator.CreateStartState());
    vnode2->Value.Set(1, 0);
    for (int action = 0; action < numAct; action++)
        if (action == 3)
            vnode2->Child(action).Value.Set(99 + numObs, 1);
        else
            vnode2->Child(action).Value.Set(100 + numAct - action, 0);
    assert(mcts.GreedyUCB(vnode2, true) == 3);

    // Action with low value and low count beats actions with high counts
    VNODE* vnode3 = mcts.ExpandNode(testSimulator.CreateStartState());
    vnode3->Value.Set(1, 0);
    for (int action = 0; action < numAct; action++)
        if (action == 3)
            vnode3->Child(action).Value.Set(1, 1);
        else
            vnode3->Child(action).Value.Set(100 + action, 1);
    assert(mcts.GreedyUCB(vnode3, true) == 3);

    // Actions with zero count is always selected
    VNODE* vnode4 = mcts.ExpandNode(testSimulator.CreateStartState());
    vnode4->Value.Set(1, 0);
    for (int action = 0; action < numAct; action++)
        if (action == 3)
            vnode4->Child(action).Value.Set(0, 0);
        else
            vnode4->Child(action).Value.Set(1, 1);
    assert(mcts.GreedyUCB(vnode4, true) == 3);
}

void MCTS::UnitTestRollout()
{
    TEST_SIMULATOR testSimulator(2, 2, 0);
    PARAMS params;
    params.NumSimulations = 1000;
    params.MaxDepth = 10;
    MCTS mcts(testSimulator, params);
    double totalReward;
    for (int n = 0; n < mcts.Params.NumSimulations; ++n)
    {
        STATE* state = testSimulator.CreateStartState();
        mcts.TreeDepth = 0;
        totalReward += mcts.Rollout(*state);
    }
    double rootValue = totalReward / mcts.Params.NumSimulations;
    double meanValue = testSimulator.MeanValue();
    assert(fabs(meanValue - rootValue) < 0.1);
}

void MCTS::UnitTestSearch(int depth)
{
    TEST_SIMULATOR testSimulator(3, 2, depth);
    PARAMS params;
    params.MaxDepth = depth + 1;
    params.NumSimulations = pow(10, depth + 1);
    MCTS mcts(testSimulator, params);
    mcts.UCTSearch();
    double rootValue = mcts.Root->Value.GetValue();
    double optimalValue = testSimulator.OptimalValue();
    assert(fabs(optimalValue - rootValue) < 0.1);
}

//-----------------------------------------------------------------------------
